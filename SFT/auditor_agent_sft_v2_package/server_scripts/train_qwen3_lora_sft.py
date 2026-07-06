import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


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


def preprocess(example, tokenizer, max_len):
    messages = example["messages"]
    full_text = apply_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = apply_template(tokenizer, messages[:2], add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False)
    prompt = tokenizer(prompt_text, add_special_tokens=False)

    input_ids = full["input_ids"][:max_len]
    attention_mask = full["attention_mask"][:max_len]
    labels = input_ids.copy()
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(x == -100 for x in labels):
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    train_file = os.path.join(args.data_dir, "train.jsonl")
    test_file = os.path.join(args.data_dir, "test.jsonl")
    ds = load_dataset("json", data_files={"train": train_file, "test": test_file})

    leak_pattern = re.compile(r"ACI_[A-Z0-9_]+|END_NEGOTIATION|success_marker|marker_check|\[Injected instruction\]")
    for split in ("train", "test"):
        for idx, row in enumerate(ds[split]):
            if leak_pattern.search(json.dumps(row, ensure_ascii=False)):
                raise ValueError(f"SFT-visible leak in {split}:{idx}")
            if not all(isinstance(msg.get("content"), str) for msg in row["messages"]):
                raise ValueError(f"Non-string message content in {split}:{idx}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = ds.map(
        lambda row: preprocess(row, tokenizer, args.max_len),
        remove_columns=ds["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
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
        bf16=True,
        tf32=True,
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
