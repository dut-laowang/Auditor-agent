"""Single-GPU Qwen3-32B QLoRA training under the frozen V19 contract."""

import argparse
import hashlib
import json
import os
import random
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

QWEN3_32B_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MARBLE_SHA256 = {
    "train": "d49ec56577a80fab3be360ae9cb1b90e2d751ce686ffa0f0e2064b2d05d0a932",
    "validation": "2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_template(tokenizer, messages, add_generation_prompt):
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def preprocess(example, tokenizer, max_len):
    messages = example["messages"]
    full_ids = tokenizer(
        apply_template(tokenizer, messages, False), add_special_tokens=False
    )["input_ids"]
    prompt_ids = tokenizer(
        apply_template(tokenizer, messages[:2], True), add_special_tokens=False
    )["input_ids"]
    boundary = 0
    for left, right in zip(full_ids, prompt_ids):
        if left != right:
            break
        boundary += 1
    prefix_ids, target_ids = full_ids[:boundary], full_ids[boundary:]
    if not target_ids or len(target_ids) >= max_len:
        raise ValueError(
            f"Invalid assistant target length={len(target_ids)} for max_len={max_len}"
        )
    total_tokens = len(prefix_ids) + len(target_ids)
    if total_tokens > max_len:
        raise ValueError(
            f"Zero-truncation V19 gate failed: sequence has {total_tokens} tokens, "
            f"exceeding max_len={max_len}"
        )
    return {
        "input_ids": prefix_ids + target_ids,
        "attention_mask": [1] * (len(prefix_ids) + len(target_ids)),
        "labels": [-100] * len(prefix_ids) + target_ids,
        "supervised_tokens": len(target_ids),
        "sequence_tokens": total_tokens,
    }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        width = max(len(item["input_ids"]) for item in features)
        output = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = width - len(item["input_ids"])
            output["input_ids"].append(
                item["input_ids"] + [self.tokenizer.pad_token_id] * pad
            )
            output["attention_mask"].append(item["attention_mask"] + [0] * pad)
            output["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--revision", default=QWEN3_32B_REVISION)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=6144)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = parser.parse_args()

    if args.model != "Qwen/Qwen3-32B" or args.revision != QWEN3_32B_REVISION:
        raise ValueError("The controlled baseline requires the pinned Qwen/Qwen3-32B revision")

    if args.max_len != 6144:
        raise ValueError("The controlled V19-32B experiment requires --max-len 6144, matching run_train_v19.sh")
    train_file = os.path.join(args.data_dir, "train.jsonl")
    validation_file = os.path.join(args.data_dir, "validation.jsonl")
    for split, path in (("train", train_file), ("validation", validation_file)):
        actual = sha256(path)
        if actual != MARBLE_SHA256[split]:
            raise ValueError(
                f"Frozen V19 MARBLE {split} hash mismatch: {actual} != {MARBLE_SHA256[split]}"
            )
    os.makedirs(args.output_dir, exist_ok=True)
    train_contract = {
        "model": args.model,
        "model_revision": args.revision,
        "train_sha256": MARBLE_SHA256["train"],
        "validation_sha256": MARBLE_SHA256["validation"],
        "max_length": args.max_len,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch": args.batch,
        "gradient_accumulation": args.grad_accum,
        "seed": args.seed,
        "quantization": args.quantization,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "targets": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
        },
    }
    contract_path = os.path.join(args.output_dir, "TRAIN_CONTRACT.json")
    if os.path.isfile(contract_path):
        with open(contract_path, encoding="utf-8") as handle:
            if json.load(handle) != train_contract:
                raise RuntimeError("Output directory contains a different Qwen3-32B training contract")
    elif get_last_checkpoint(args.output_dir):
        raise RuntimeError("Existing checkpoints have no TRAIN_CONTRACT.json; use a fresh output directory")
    else:
        with open(contract_path, "w", encoding="utf-8") as handle:
            json.dump(train_contract, handle, ensure_ascii=False, indent=2)
    ds = load_dataset("json", data_files={"train": train_file, "validation": validation_file})
    leak_pattern = re.compile(
        r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|success_marker|success_markers|"
        r"attack_metadata|attack_id|marker_check|\[Injected[^\]]*\]|"
        r"offline verifier|attack-success index|labeled as attack-success",
        re.I,
    )
    for split in ("train", "validation"):
        for idx, row in enumerate(ds[split]):
            if leak_pattern.search(json.dumps(row["messages"][:2], ensure_ascii=False)):
                raise ValueError(f"SFT-visible leak in {split}:{idx}")
            if len(row["messages"]) != 3 or [m["role"] for m in row["messages"]] != [
                "system", "user", "assistant"
            ]:
                raise ValueError(f"Unexpected V19 message schema in {split}:{idx}")

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
        lambda row: preprocess(row, tokenizer, args.max_len),
        remove_columns=ds["train"].column_names,
    )
    for split in ("train", "validation"):
        supervised = ds[split]["supervised_tokens"]
        sequence_tokens = ds[split]["sequence_tokens"]
        if min(supervised) < 16:
            raise ValueError(f"Assistant supervision too short in {split}")
        print(json.dumps({
            "preprocess_split": split,
            "rows": len(supervised),
            "min_supervised_tokens": min(supervised),
            "max_supervised_tokens": max(supervised),
            "min_sequence_tokens": min(sequence_tokens),
            "max_sequence_tokens": max(sequence_tokens),
            "prompt_truncated_rows": 0,
        }))
    ds = ds.remove_columns(["supervised_tokens", "sequence_tokens"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None and model_context < args.max_len:
        raise ValueError(f"Model context limit {model_context} < {args.max_len}")
    model.config.use_cache = False
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    ))
    model.print_trainable_parameters()

    longest = sorted(
        range(len(ds["train"])),
        key=lambda index: len(ds["train"][index]["input_ids"]),
        reverse=True,
    )[: args.batch]
    smoke_batch = DataCollator(tokenizer)([ds["train"][index] for index in longest])
    smoke_batch = {key: value.to(model.device) for key, value in smoke_batch.items()}
    model.train()
    smoke_loss = model(**smoke_batch).loss
    if not torch.isfinite(smoke_loss):
        raise RuntimeError(f"Non-finite Qwen3-32B smoke loss: {smoke_loss.item()}")
    smoke_loss.backward()
    smoke_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not smoke_gradients or not all(torch.isfinite(gradient).all() for gradient in smoke_gradients):
        raise RuntimeError("Qwen3-32B smoke backward produced missing or non-finite LoRA gradients")
    if not any(torch.count_nonzero(gradient).item() for gradient in smoke_gradients):
        raise RuntimeError("Qwen3-32B smoke backward produced only zero LoRA gradients")
    model.zero_grad(set_to_none=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(json.dumps({
        "runtime_smoke": "PASS",
        "rows": longest,
        "batch": len(longest),
        "max_sequence_tokens": max(len(ds["train"][index]["input_ids"]) for index in longest),
        "loss": float(smoke_loss.detach().cpu()),
    }))
    del smoke_loss, smoke_batch, smoke_gradients
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=torch.cuda.get_device_capability()[0] >= 8,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.quantization == "4bit" else "adamw_torch_fused",
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
    checkpoint = None
    if args.resume == "auto" and os.path.isdir(args.output_dir):
        checkpoint = get_last_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    adapter_artifacts = {
        name: sha256(os.path.join(args.output_dir, name))
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
        if os.path.isfile(os.path.join(args.output_dir, name))
    }
    if not adapter_artifacts:
        raise RuntimeError("Training finished without saved LoRA adapter artifacts")
    manifest = {
        "version": "V19-qualityfix-qwen3-32b",
        "controlled_against": "Qwen3-8B V19",
        "model": args.model,
        "model_revision": args.revision,
        "quantization": args.quantization,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch": args.batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "max_length": args.max_len,
        "train_sha256": sha256(train_file),
        "validation_sha256": sha256(validation_file),
        "test_accessed": False,
        "resume_from_checkpoint": checkpoint,
        "adapter_artifacts": adapter_artifacts,
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
