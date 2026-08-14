import argparse
import hashlib
import json
import os
import random
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VERDICTS = ("clean_safe", "attack_failed", "attack_success")


def apply_template(tokenizer, messages, add_generation_prompt):
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def localization_span(content):
    start = content.find('"localization"')
    if start < 0:
        raise ValueError("Assistant target has no localization field")
    object_start = content.find("{", content.find(":", start))
    if object_start < 0:
        raise ValueError("Malformed localization object")
    depth = 0
    quoted = False
    escaped = False
    for index in range(object_start, len(content)):
        char = content[index]
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("Unterminated localization object")


def preprocess(row, tokenizer, max_len):
    messages = row["messages"]
    target = json.loads(messages[2]["content"])
    verdict = target["decision"]["verdict"]
    if verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict: {verdict}")

    full_text = apply_template(tokenizer, messages, False)
    prompt_text = apply_template(tokenizer, messages[:2], True)
    full = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt = tokenizer(prompt_text, add_special_tokens=False)
    full_ids = full["input_ids"]
    prompt_ids = prompt["input_ids"]

    boundary = 0
    for left, right in zip(full_ids, prompt_ids):
        if left != right:
            break
        boundary += 1
    if boundary == 0 or boundary >= len(full_ids):
        raise ValueError("Invalid assistant boundary")
    if len(full_ids) > max_len:
        raise ValueError(f"Zero-truncation gate failed: {len(full_ids)} > {max_len}")

    content = messages[2]["content"]
    content_start = full_text.rfind(content)
    if content_start < 0:
        raise ValueError("Assistant content not found verbatim in rendered conversation")
    loc_start, loc_end = localization_span(content)
    loc_start += content_start
    loc_end += content_start

    labels = [-100] * boundary + full_ids[boundary:]
    loc_mask = [0] * len(full_ids)
    for index, (start, end) in enumerate(full["offset_mapping"]):
        if index >= boundary and end > loc_start and start < loc_end:
            loc_mask[index] = 1
    if not any(loc_mask):
        raise ValueError("Localization token mask is empty")

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "localization_mask": loc_mask,
        "prompt_index": boundary - 1,
        "verdict_label": VERDICTS.index(verdict),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - boundary,
    }


class Collator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels", "localization_mask")}
        for row in rows:
            padding = width - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
            batch["localization_mask"].append(row["localization_mask"] + [0] * padding)
        output = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        output["prompt_index"] = torch.tensor([row["prompt_index"] for row in rows], dtype=torch.long)
        output["verdict_label"] = torch.tensor([row["verdict_label"] for row in rows], dtype=torch.long)
        return output


class JointTrainer(Trainer):
    def __init__(self, *args, lambda_cls, lambda_loc, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_index = inputs.pop("prompt_index")
        verdict_label = inputs.pop("verdict_label")
        localization_mask = inputs.pop("localization_mask")
        labels = inputs["labels"]
        # Call the decoder backbone directly so only its final state is retained.
        # output_hidden_states=True would keep all 36 long-context layer states
        # and needlessly increase H100 memory consumption.
        causal_lm = model.get_base_model()
        decoder = causal_lm.model
        decoder_outputs = decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = decoder_outputs.last_hidden_state
        logits = causal_lm.lm_head(hidden)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        weights = (shift_labels != -100).to(token_loss.dtype)
        loc = localization_mask[:, 1:].to(token_loss.dtype)
        weights = weights * (1.0 + (self.lambda_loc - 1.0) * loc)
        report_loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

        rows = torch.arange(hidden.size(0), device=hidden.device)
        prompt_repr = hidden[rows, prompt_index.to(hidden.device)]
        verdict_logits = model.verdict_head(prompt_repr.float())
        class_loss = F.cross_entropy(verdict_logits, verdict_label.to(verdict_logits.device))
        loss = report_loss + self.lambda_cls * class_loss
        outputs = {"loss": loss, "logits": logits}
        return (loss, outputs) if return_outputs else loss


class HeadCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "verdict_head.pt")
        torch.save(model.verdict_head.state_dict(), path)
        return control


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-cls", type=float, default=0.5)
    parser.add_argument("--lambda-loc", type=float, default=2.0)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()
    if args.max_len != 8192 or args.lambda_cls <= 0 or args.lambda_loc < 1:
        raise ValueError("Invalid frozen joint-SFT contract")

    files = {split: os.path.join(args.data_dir, f"{split}.jsonl") for split in ("train", "validation")}
    dataset = load_dataset("json", data_files=files)
    leak = re.compile(
        r"ACI_[A-Z0-9_]+|\\baci_[a-z0-9_]+\\b|END_NEGOTIATION|success_marker|"
        r"attack_metadata|attack_id|marker_check|\\[Injected[^\\]]*\\]|offline verifier|"
        r"attack-success index|labeled as attack-success",
        re.I,
    )
    for split in dataset:
        for index, row in enumerate(dataset[split]):
            if [m.get("role") for m in row["messages"]] != ["system", "user", "assistant"]:
                raise ValueError(f"Bad role contract at {split}:{index}")
            visible = json.dumps(row["messages"][:2], ensure_ascii=False)
            if leak.search(visible):
                raise ValueError(f"SFT-visible leak at {split}:{index}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = dataset.map(
        lambda row: preprocess(row, tokenizer, args.max_len),
        remove_columns=dataset["train"].column_names,
    )
    for split in dataset:
        print(json.dumps({
            "split": split,
            "rows": len(dataset[split]),
            "min_tokens": min(dataset[split]["sequence_tokens"]),
            "max_tokens": max(dataset[split]["sequence_tokens"]),
            "target_source": "assistant_only_not_model_input",
        }))
    dataset = dataset.remove_columns(["sequence_tokens", "supervised_tokens"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    tf32 = torch.cuda.get_device_capability()[0] >= 8
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    hidden = model.get_base_model().config.hidden_size
    model.add_module("verdict_head", torch.nn.Linear(hidden, len(VERDICTS), dtype=torch.float32))
    model.verdict_head.to(model.device)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        bf16=bf16,
        fp16=not bf16,
        tf32=tf32,
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
    trainer = JointTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=Collator(tokenizer),
        lambda_cls=args.lambda_cls,
        lambda_loc=args.lambda_loc,
        callbacks=[HeadCheckpoint()],
    )
    smoke = Collator(tokenizer)([dataset["train"][0]])
    smoke = {key: value.to(model.device) for key, value in smoke.items()}
    with torch.no_grad():
        smoke_loss = trainer.compute_loss(model, smoke)
    if not torch.isfinite(smoke_loss):
        raise RuntimeError("Joint-loss forward preflight is not finite")
    print(json.dumps({"joint_forward_preflight": "pass", "loss": float(smoke_loss)}))
    checkpoint = get_last_checkpoint(args.output_dir) if args.resume == "auto" else None
    if checkpoint:
        head_file = os.path.join(checkpoint, "verdict_head.pt")
        if not os.path.isfile(head_file):
            raise RuntimeError(f"Joint checkpoint lacks verdict head: {head_file}")
        model.verdict_head.load_state_dict(torch.load(head_file, map_location="cpu", weights_only=True))
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.verdict_head.state_dict(), os.path.join(args.output_dir, "verdict_head.pt"))
    manifest = {
        "version": "V20-AppWorld-MARBLE-joint-audit-v1",
        "model": args.model,
        "model_revision": args.revision,
        "train_sha256": sha256(files["train"]),
        "validation_sha256": sha256(files["validation"]),
        "test_accessed": False,
        "input_contract": "messages[0:2] only",
        "output_contract": "unchanged V20 full JSON via LM head",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch": args.batch,
        "gradient_accumulation": args.grad_accum,
        "max_length": args.max_len,
        "lambda_classification": args.lambda_cls,
        "lambda_localization_token_weight": args.lambda_loc,
        "classification_head_inference": False,
        "localization_head_forced": False,
    }
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
