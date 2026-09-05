import argparse
import hashlib
import json
import os
import re
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def apply_template(tokenizer, messages, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def preprocess(example, tokenizer, max_len, prompt_overflow="error"):
    messages = example["messages"]
    full_text = apply_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = apply_template(tokenizer, messages[:2], add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False)
    prompt = tokenizer(prompt_text, add_special_tokens=False)

    full_ids = full["input_ids"]
    prompt_ids = prompt["input_ids"]
    # The generation-prompt template and the full conversation can differ at
    # the assistant boundary. Find their actual common prefix instead of
    # assuming prompt length is an exact boundary.
    boundary = 0
    for left, right in zip(full_ids, prompt_ids):
        if left != right:
            break
        boundary += 1
    prefix_ids = full_ids[:boundary]
    target_ids = full_ids[boundary:]
    if not target_ids:
        raise ValueError("No assistant target tokens after chat-template boundary")
    if len(target_ids) >= max_len:
        raise ValueError(
            f"Assistant target has {len(target_ids)} tokens, exceeding max_len={max_len}"
        )

    total_tokens = len(prefix_ids) + len(target_ids)
    if total_tokens > max_len:
        if prompt_overflow == "error":
            raise ValueError(
                f"Zero-truncation V19 gate failed: sequence has {total_tokens} tokens, "
                f"exceeding max_len={max_len}"
            )
        # V20 contains a small number of substantially longer trajectories than
        # V19. Preserve the complete assistant target plus both ends of the
        # prompt: the task/schema at the front and final outcome evidence at the
        # back. Only the middle of an overlength prompt is removed.
        keep_prefix = max_len - len(target_ids)
        keep_head = (keep_prefix + 1) // 2
        keep_tail = keep_prefix - keep_head
        prefix_ids = prefix_ids[:keep_head] + (prefix_ids[-keep_tail:] if keep_tail else [])
        total_tokens = len(prefix_ids) + len(target_ids)
    input_ids = prefix_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prefix_ids) + target_ids
    if not any(token != -100 for token in labels):
        raise ValueError("Assistant supervision was lost during preprocessing")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "supervised_tokens": len(target_ids),
        "sequence_tokens": total_tokens,
        "prompt_truncated": int(len(full_ids) > max_len),
    }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            out["attention_mask"].append(item["attention_mask"] + [0] * pad)
            out["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=QWEN3_8B_REVISION)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument(
        "--context-contract",
        choices=["v19-8192", "v22-all-12288", "v23-all-12288", "v23-internlm-12800"],
        default="v19-8192",
        help="Explicit experiment context contract; V19 remains fixed at 8192 by default.",
    )
    parser.add_argument(
        "--prompt-overflow",
        choices=["error", "middle"],
        default="error",
        help="Keep V19 strict by default; V20 may middle-truncate only an overlength prompt.",
    )
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-cudnn-sdp",
        action="store_true",
        help="Disable the cuDNN SDPA backend while retaining other PyTorch SDPA backends.",
    )
    parser.add_argument("--init-adapter", help="Optional existing LoRA adapter to continue training, e.g. v2 adapter.")
    parser.add_argument(
        "--resume",
        choices=["auto", "never"],
        default="auto",
        help="Automatically resume from the newest checkpoint in output-dir.",
    )
    args = parser.parse_args()
    if args.disable_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    required_max_len = {
        "v19-8192": 8192,
        "v22-all-12288": 12288,
        "v23-all-12288": 12288,
        "v23-internlm-12800": 12800,
    }[args.context_contract]
    if args.max_len != required_max_len:
        raise ValueError(
            f"Context contract {args.context_contract} requires --max-len {required_max_len}"
        )

    train_file = os.path.join(args.data_dir, "train.jsonl")
    validation_file = os.path.join(args.data_dir, "validation.jsonl")
    ds = load_dataset(
        "json", data_files={"train": train_file, "validation": validation_file}
    )

    leak_pattern = re.compile(
        r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|success_marker|success_markers|"
        r"attack_metadata|attack_id|marker_check|\[Injected[^\]]*\]|"
        r"offline verifier|attack-success index|labeled as attack-success",
        re.I,
    )
    for split in ("train", "validation"):
        for idx, row in enumerate(ds[split]):
            visible = json.dumps(row["messages"][:2], ensure_ascii=False)
            if leak_pattern.search(visible):
                raise ValueError(f"SFT-visible leak in {split}:{idx}")
            if not all(isinstance(msg.get("content"), str) for msg in row["messages"]):
                raise ValueError(f"Non-string message content in {split}:{idx}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.model_max_length < args.max_len:
        raise ValueError(
            f"Tokenizer context limit {tokenizer.model_max_length} < {args.max_len}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = ds.map(
        lambda row: preprocess(row, tokenizer, args.max_len, args.prompt_overflow),
        remove_columns=ds["train"].column_names,
    )
    for split in ("train", "validation"):
        supervised = ds[split]["supervised_tokens"]
        sequence_tokens = ds[split]["sequence_tokens"]
        prompt_truncated = ds[split]["prompt_truncated"]
        if not supervised or min(supervised) < 16:
            raise ValueError(
                f"Invalid assistant supervision in {split}: "
                f"minimum target tokens={min(supervised) if supervised else 0}"
            )
        print(
            json.dumps(
                {
                    "preprocess_split": split,
                    "rows": len(supervised),
                    "min_supervised_tokens": min(supervised),
                    "max_supervised_tokens": max(supervised),
                    "min_sequence_tokens": min(sequence_tokens),
                    "max_sequence_tokens": max(sequence_tokens),
                    "prompt_truncated_rows": sum(prompt_truncated),
                }
            )
        )
    ds = ds.remove_columns(["supervised_tokens", "sequence_tokens", "prompt_truncated"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for V19 training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    use_tf32 = torch.cuda.get_device_capability()[0] >= 8
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        json.dumps(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "precision": "bf16" if use_bf16 else "fp16",
                "tf32": use_tf32,
                "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            }
        )
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None and model_context < args.max_len:
        raise ValueError(f"Model context limit {model_context} < {args.max_len}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=use_tf32,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollator(tokenizer),
    )
    resume_checkpoint = None
    if args.resume == "auto" and os.path.isdir(args.output_dir):
        resume_checkpoint = get_last_checkpoint(args.output_dir)
    print(
        json.dumps(
            {
                "resume_policy": args.resume,
                "resume_from_checkpoint": resume_checkpoint,
            }
        )
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    run_manifest = {
        "version": {
            "v19-8192": "V19-qualityfix",
            "v22-all-12288": "V22-ALL-audit-grade-sft-v1",
            "v23-all-12288": "V23-ALL-audit-grade-sft-v1",
            "v23-internlm-12800": "V23-ALL-audit-grade-sft-v1",
        }[args.context_contract],
        "model": args.model,
        "model_revision": args.revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch": args.batch,
        "gradient_accumulation": args.grad_accum,
        "max_length": args.max_len,
        "context_contract": args.context_contract,
        "prompt_overflow": args.prompt_overflow,
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "train_sha256": sha256(train_file),
        "validation_sha256": sha256(validation_file),
        "test_accessed": False,
        "resume_from_checkpoint": resume_checkpoint,
        "init_adapter": os.path.abspath(args.init_adapter) if args.init_adapter else None,
        "adapter_artifacts": {
            name: sha256(os.path.join(args.output_dir, name))
            for name in sorted(os.listdir(args.output_dir))
            if os.path.isfile(os.path.join(args.output_dir, name))
            and name != "run_manifest.json"
        },
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
