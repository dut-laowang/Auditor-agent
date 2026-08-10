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

    if args.max_len != 6144:
        raise ValueError("The controlled V19-32B experiment requires --max-len 6144, matching run_train_v19.sh")
    train_file = os.path.join(args.data_dir, "train.jsonl")
    validation_file = os.path.join(args.data_dir, "validation.jsonl")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
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
    manifest = {
        "version": "V19-qualityfix-qwen3-32b",
        "controlled_against": "Qwen3-8B V19",
        "model": args.model,
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
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
