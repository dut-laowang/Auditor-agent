"""ModernBERT-4096 multi-task baseline for the frozen V19 MARBLE split."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

VERDICTS = ["clean_safe", "attack_failed", "attack_success"]
SCOPES = ["none", "global", "node", "edge", "tool", "multi"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_target(row):
    target = json.loads(row["messages"][2]["content"])
    verdict = target["decision"]["verdict"]
    localization = target.get("localization", {})
    return verdict, str(localization.get("scope", "none")), {
        str(value) for value in localization.get("component_ids", [])
    }


def visible_text(row, input_mode):
    system = row["messages"][0]["content"]
    user = row["messages"][1]["content"]
    if input_mode == "user":
        return user
    return f"[SYSTEM]\n{system}\n[USER]\n{user}"


def candidates_from_row(row):
    user = json.loads(row["messages"][1]["content"])
    candidates = []
    seen = set()
    for candidate in user.get("graph_candidates", []):
        if not isinstance(candidate, dict) or not candidate.get("id"):
            continue
        component_id = str(candidate["id"])
        if component_id in seen:
            continue
        seen.add(component_id)
        candidates.append((component_id, json.dumps(candidate, ensure_ascii=False, sort_keys=True)))
    return candidates


def validate_contract(rows, split):
    missing = []
    for idx, row in enumerate(rows):
        if [message.get("role") for message in row.get("messages", [])] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"Unexpected message schema at {split}:{idx}")
        verdict, scope, gold = parse_target(row)
        if verdict not in VERDICTS or scope not in SCOPES:
            raise ValueError(f"Unknown target at {split}:{idx}: {verdict}/{scope}")
        candidate_ids = {item[0] for item in candidates_from_row(row)}
        if not gold.issubset(candidate_ids):
            missing.append((idx, sorted(gold - candidate_ids)))
    if missing:
        raise ValueError(
            f"Gold components absent from graph_candidates in {split}; first={missing[:3]}"
        )


class V19Dataset(Dataset):
    def __init__(self, rows, tokenizer, max_len, candidate_max_len, input_mode):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.candidate_max_len = candidate_max_len
        self.input_mode = input_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        verdict, scope, gold = parse_target(row)
        candidates = candidates_from_row(row)
        document = self.tokenizer(
            visible_text(row, self.input_mode),
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )["input_ids"]
        candidate_ids = [item[0] for item in candidates]
        candidate_tokens = [
            self.tokenizer(
                text,
                truncation=True,
                max_length=self.candidate_max_len,
                add_special_tokens=True,
            )["input_ids"]
            for _, text in candidates
        ]
        return {
            "run_id": row.get("metadata", {}).get("run_id"),
            "document": document,
            "candidates": candidate_tokens,
            "candidate_ids": candidate_ids,
            "component_labels": [float(item in gold) for item in candidate_ids],
            "verdict": VERDICTS.index(verdict),
            "scope": SCOPES.index(scope),
        }


class Collator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def _pad(self, sequences):
        width = max(map(len, sequences))
        ids, masks = [], []
        for sequence in sequences:
            amount = width - len(sequence)
            ids.append(sequence + [self.pad] * amount)
            masks.append([1] * len(sequence) + [0] * amount)
        return torch.tensor(ids), torch.tensor(masks)

    def __call__(self, items):
        doc_ids, doc_mask = self._pad([item["document"] for item in items])
        flat_candidates, owners, labels = [], [], []
        candidate_ids = []
        for owner, item in enumerate(items):
            flat_candidates.extend(item["candidates"])
            owners.extend([owner] * len(item["candidates"]))
            labels.extend(item["component_labels"])
            candidate_ids.append(item["candidate_ids"])
        if not flat_candidates:
            raise ValueError("V19 row has no graph candidates")
        candidate_input_ids, candidate_attention_mask = self._pad(flat_candidates)
        return {
            "run_ids": [item["run_id"] for item in items],
            "input_ids": doc_ids,
            "attention_mask": doc_mask,
            "candidate_input_ids": candidate_input_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "candidate_owner": torch.tensor(owners),
            "candidate_labels": torch.tensor(labels, dtype=torch.float32),
            "candidate_ids": candidate_ids,
            "verdict_labels": torch.tensor([item["verdict"] for item in items]),
            "scope_labels": torch.tensor([item["scope"] for item in items]),
        }


class ModernBertMultiTask(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "classifier_dropout", None) or 0.1
        self.dropout = nn.Dropout(dropout)
        self.verdict_head = nn.Linear(hidden, len(VERDICTS))
        self.scope_head = nn.Linear(hidden, len(SCOPES))
        self.component_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    @staticmethod
    def pool(output):
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]

    def forward(self, batch):
        document = self.pool(self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ))
        candidate = self.pool(self.encoder(
            input_ids=batch["candidate_input_ids"],
            attention_mask=batch["candidate_attention_mask"],
        ))
        document = self.dropout(document)
        owned_document = document[batch["candidate_owner"]]
        pair = torch.cat(
            [owned_document, candidate, torch.abs(owned_document - candidate), owned_document * candidate],
            dim=-1,
        )
        return (
            self.verdict_head(document),
            self.scope_head(document),
            self.component_head(pair).squeeze(-1),
        )


def move(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predict(model, loader, device, threshold):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="modernbert_eval"):
            batch = move(batch, device)
            verdict_logits, scope_logits, component_logits = model(batch)
            verdict_pred = verdict_logits.argmax(-1).cpu().tolist()
            scope_pred = scope_logits.argmax(-1).cpu().tolist()
            probabilities = torch.sigmoid(component_logits).cpu().tolist()
            owners = batch["candidate_owner"].cpu().tolist()
            gold_component_flat = batch["candidate_labels"].cpu().tolist()
            for row_index, run_id in enumerate(batch["run_ids"]):
                positions = [i for i, owner in enumerate(owners) if owner == row_index]
                ids = batch["candidate_ids"][row_index]
                gold_components = [ids[j] for j, pos in enumerate(positions) if gold_component_flat[pos] > 0.5]
                pred_components = [ids[j] for j, pos in enumerate(positions) if probabilities[pos] >= threshold]
                records.append({
                    "run_id": run_id,
                    "gold": VERDICTS[batch["verdict_labels"][row_index].item()],
                    "pred": VERDICTS[verdict_pred[row_index]],
                    "gold_scope": SCOPES[batch["scope_labels"][row_index].item()],
                    "pred_scope": SCOPES[scope_pred[row_index]],
                    "gold_components": sorted(gold_components),
                    "pred_components": sorted(pred_components),
                    "component_ids": ids,
                    "component_probabilities": [probabilities[pos] for pos in positions],
                })
    return records


def component_f1(records, threshold=None):
    tp = fp = fn = 0
    for row in records:
        gold = set(row["gold_components"])
        if threshold is None:
            pred = set(row["pred_components"])
        else:
            pred = {
                component_id
                for component_id, probability in zip(
                    row["component_ids"], row["component_probabilities"]
                )
                if probability >= threshold
            }
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tune_threshold(records):
    records = [
        row
        for row in records
        if row["gold"] == "attack_success" and row["gold_components"]
    ]
    choices = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
    return max(choices, key=lambda value: (component_f1(records, value), -abs(value - 0.5)))


def metrics_from_records(records, threshold):
    for row in records:
        row["pred_components"] = sorted(
            component_id
            for component_id, probability in zip(
                row["component_ids"], row["component_probabilities"]
            )
            if probability >= threshold
        )
    y3, p3 = [r["gold"] for r in records], [r["pred"] for r in records]
    binary = lambda value: "unsafe" if value == "attack_success" else "safe"
    yb, pb = [binary(v) for v in y3], [binary(v) for v in p3]
    localized = [r for r in records if r["gold"] == "attack_success" and r["gold_components"]]
    tp = sum(len(set(r["gold_components"]) & set(r["pred_components"])) for r in localized)
    fp = sum(len(set(r["pred_components"]) - set(r["gold_components"])) for r in localized)
    fn = sum(len(set(r["gold_components"]) - set(r["pred_components"])) for r in localized)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "mode": "modernbert_multitask",
        "n": len(records),
        "component_threshold": threshold,
        "gold_distribution": dict(Counter(y3)),
        "prediction_distribution": dict(Counter(p3)),
        "three_class_accuracy": accuracy_score(y3, p3),
        "three_class_report": classification_report(y3, p3, labels=VERDICTS, zero_division=0, output_dict=True),
        "binary_accuracy": accuracy_score(yb, pb),
        "binary_report": classification_report(yb, pb, labels=["safe", "unsafe"], zero_division=0, output_dict=True),
        "localization": {
            "n_attack_success_with_gold_components": len(localized),
            "component_micro_precision": precision,
            "component_micro_recall": recall,
            "component_micro_f1": f1,
            "component_hit_rate": sum(bool(set(r["gold_components"]) & set(r["pred_components"])) for r in localized) / len(localized) if localized else 0.0,
            "component_exact_match": sum(set(r["gold_components"]) == set(r["pred_components"]) for r in localized) / len(localized) if localized else 0.0,
            "scope_accuracy": sum(r["gold_scope"] == r["pred_scope"] for r in localized) / len(localized) if localized else 0.0,
            "localization_policy": "source attack-placement candidate projection",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-role", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--sealed-test-ack", choices=["FINAL_ONCE"])
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--candidate-max-len", type=int, default=256)
    parser.add_argument("--input-mode", choices=["user", "system_user"], default="user")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lambda-scope", type=float, default=1.0)
    parser.add_argument("--lambda-component", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_len != 4096:
        raise ValueError("The controlled ModernBERT experiment requires --max-len 4096")
    if args.dataset_role == "test" and args.sealed_test_ack != "FINAL_ONCE":
        raise ValueError("Final test requires --sealed-test-ack FINAL_ONCE")
    if args.mode == "train" and args.dataset_role != "train":
        raise ValueError("Training requires --dataset-role train")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data_file)
    validate_contract(rows, args.dataset_role)
    component_positive = sum(len(parse_target(row)[2]) for row in rows)
    component_total = sum(len(candidates_from_row(row)) for row in rows)
    component_pos_weight = max(
        1.0,
        (component_total - component_positive) / max(1, component_positive),
    )
    loader = DataLoader(
        V19Dataset(rows, tokenizer, args.max_len, args.candidate_max_len, args.input_mode),
        batch_size=args.batch,
        shuffle=args.mode == "train",
        collate_fn=Collator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = ModernBertMultiTask(args.model).to(device)

    if args.mode == "train":
        model.encoder.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        updates = (len(loader) + args.grad_accum - 1) // args.grad_accum * args.epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(updates * 0.03)), updates)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for epoch in range(args.epochs):
            progress = tqdm(loader, desc=f"modernbert_train_epoch_{epoch + 1}")
            for batch_index, batch in enumerate(progress):
                batch = move(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    verdict_logits, scope_logits, component_logits = model(batch)
                    verdict_loss = F.cross_entropy(verdict_logits, batch["verdict_labels"])
                    scope_loss = F.cross_entropy(scope_logits, batch["scope_labels"])
                    component_loss = F.binary_cross_entropy_with_logits(
                        component_logits,
                        batch["candidate_labels"],
                        pos_weight=torch.tensor(component_pos_weight, device=device),
                    )
                    loss = (verdict_loss + args.lambda_scope * scope_loss + args.lambda_component * component_loss) / args.grad_accum
                scaler.scale(loss).backward()
                if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1
                progress.set_postfix(loss=float(loss.item() * args.grad_accum))
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}.pt"))
        tokenizer.save_pretrained(args.output_dir)
        config = vars(args).copy()
        config.update({
            "train_sha256": sha256(args.data_file),
            "optimizer_updates": step,
            "component_positive": component_positive,
            "component_total_candidates": component_total,
            "component_pos_weight": component_pos_weight,
            "test_accessed": False,
        })
        with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        return

    if not args.checkpoint:
        raise ValueError("Evaluation requires --checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    threshold = args.threshold
    if threshold is None:
        threshold_file = os.path.join(os.path.dirname(args.checkpoint), "component_threshold.json")
        if os.path.isfile(threshold_file):
            threshold = json.load(open(threshold_file, encoding="utf-8"))["threshold"]
        elif args.dataset_role == "validation":
            threshold = 0.5
        else:
            raise ValueError("Test requires a frozen validation threshold")
    if args.dataset_role == "test":
        seal_path = os.path.join(args.output_dir, "SEALED_TEST_CONSUMED.json")
        if os.path.exists(seal_path):
            raise RuntimeError(f"Sealed test already consumed: {seal_path}")
        with open(seal_path, "w", encoding="utf-8") as handle:
            json.dump({"test_sha256": sha256(args.data_file), "rows": len(rows)}, handle, indent=2)
    records = predict(model, loader, device, threshold)
    if args.dataset_role == "validation" and args.threshold is None:
        threshold = tune_threshold(records)
        with open(os.path.join(os.path.dirname(args.checkpoint), "component_threshold.json"), "w", encoding="utf-8") as handle:
            json.dump({"threshold": threshold, "selected_on": "validation", "validation_sha256": sha256(args.data_file)}, handle, indent=2)
    metrics = metrics_from_records(records, threshold)
    metrics.update({
        "model": args.model,
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "data_sha256": sha256(args.data_file),
        "dataset_role": args.dataset_role,
        "max_length": args.max_len,
        "input_mode": args.input_mode,
    })
    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
