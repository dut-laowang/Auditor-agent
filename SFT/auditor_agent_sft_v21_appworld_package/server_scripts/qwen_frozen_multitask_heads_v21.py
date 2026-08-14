from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VERDICTS = ("clean_safe", "attack_failed", "attack_success")
SCOPES = ("none", "global", "node", "edge", "tool", "multi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def apply_template(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def final_state(model, tokenizer, texts, max_len, batch_size, desc):
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > max_len:
            raise ValueError(f"Zero-truncation feature gate failed: {encoded['input_ids'].shape[1]} > {max_len}")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.get_base_model().model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        vectors.extend(output.last_hidden_state[:, -1, :].to(torch.float16).cpu())
    return vectors


def build_cache(data_file, cache_file, model_name, revision, adapter, max_len, candidate_batch):
    rows = read(data_file)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompts = [apply_template(tokenizer, row["messages"]) for row in rows]
    document_vectors = final_state(model, tokenizer, prompts, max_len, 1, "qwen_document_features")
    cached_rows = []
    for index, row in enumerate(tqdm(rows, desc="qwen_candidate_features")):
        candidates = row["graph_candidates"]
        texts = [
            apply_template(tokenizer, [{"role": "user", "content": "Graph candidate:\n" + json.dumps(item, ensure_ascii=False, sort_keys=True)}])
            for item in candidates
        ]
        candidate_vectors = final_state(
            model, tokenizer, texts, 512, candidate_batch, "candidate_batch"
        )
        gold = row["target"]
        ids = [str(item["id"]) for item in candidates]
        cached_rows.append({
            "run_id": row["run_id"],
            "document": document_vectors[index],
            "candidates": torch.stack(candidate_vectors),
            "candidate_ids": ids,
            "candidate_labels": torch.tensor(
                [float(value in set(gold["component_ids"])) for value in ids], dtype=torch.float32
            ),
            "verdict": VERDICTS.index(gold["verdict"]),
            "scope": SCOPES.index(gold["scope"]),
        })
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "data_sha256": sha256(data_file),
        "adapter_manifest_sha256": sha256(Path(adapter) / "run_manifest.json"),
        "hidden_size": int(cached_rows[0]["document"].numel()),
        "rows": cached_rows,
    }, cache_file)
    del model, base
    torch.cuda.empty_cache()


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.payload = torch.load(path, map_location="cpu", weights_only=False)
        self.rows = self.payload["rows"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(items):
    candidates, owners, labels, ids = [], [], [], []
    for owner, item in enumerate(items):
        candidates.append(item["candidates"])
        owners.extend([owner] * len(item["candidate_ids"]))
        labels.append(item["candidate_labels"])
        ids.append(item["candidate_ids"])
    return {
        "run_ids": [item["run_id"] for item in items],
        "document": torch.stack([item["document"] for item in items]).float(),
        "candidates": torch.cat(candidates).float(),
        "owners": torch.tensor(owners),
        "candidate_labels": torch.cat(labels),
        "candidate_ids": ids,
        "verdict": torch.tensor([item["verdict"] for item in items]),
        "scope": torch.tensor([item["scope"] for item in items]),
    }


class Heads(nn.Module):
    def __init__(self, hidden, width=512, dropout=0.1):
        super().__init__()
        self.verdict = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 3))
        self.scope = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 6))
        self.component = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )

    def forward(self, batch):
        document = batch["document"]
        candidate = batch["candidates"]
        owned = document[batch["owners"]]
        pair = torch.cat([owned, candidate, torch.abs(owned - candidate), owned * candidate], dim=-1)
        return self.verdict(document), self.scope(document), self.component(pair).squeeze(-1)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def records(model, loader, device):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            verdict, scope, component = model(batch)
            vpred = verdict.argmax(-1).cpu().tolist()
            spred = scope.argmax(-1).cpu().tolist()
            probs = torch.sigmoid(component).cpu().tolist()
            owners = batch["owners"].cpu().tolist()
            labels = batch["candidate_labels"].cpu().tolist()
            for row_index, run_id in enumerate(batch["run_ids"]):
                positions = [i for i, owner in enumerate(owners) if owner == row_index]
                ids = batch["candidate_ids"][row_index]
                output.append({
                    "run_id": run_id,
                    "gold": VERDICTS[batch["verdict"][row_index].item()],
                    "pred": VERDICTS[vpred[row_index]],
                    "gold_scope": SCOPES[batch["scope"][row_index].item()],
                    "pred_scope": SCOPES[spred[row_index]],
                    "gold_components": [ids[j] for j, pos in enumerate(positions) if labels[pos] > 0.5],
                    "pred_components": [],
                    "component_ids": ids,
                    "component_probabilities": [probs[pos] for pos in positions],
                })
    return output


def component_f1(rows, threshold):
    tp = fp = fn = 0
    for row in rows:
        if row["gold"] != "attack_success" or not row["gold_components"]:
            continue
        gold = set(row["gold_components"])
        pred = {cid for cid, prob in zip(row["component_ids"], row["component_probabilities"]) if prob >= threshold}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def threshold(rows):
    choices = [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]
    return max(choices, key=lambda value: (component_f1(rows, value), -abs(value - 0.5)))


def metrics(rows, cutoff):
    for row in rows:
        row["pred_components"] = [
            cid for cid, prob in zip(row["component_ids"], row["component_probabilities"]) if prob >= cutoff
        ]
    y = [row["gold"] for row in rows]; p = [row["pred"] for row in rows]
    localized = [row for row in rows if row["gold"] == "attack_success" and row["gold_components"]]
    tp = sum(len(set(r["gold_components"]) & set(r["pred_components"])) for r in localized)
    fp = sum(len(set(r["pred_components"]) - set(r["gold_components"])) for r in localized)
    fn = sum(len(set(r["gold_components"]) - set(r["pred_components"])) for r in localized)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows), "component_threshold": cutoff,
        "three_class_accuracy": accuracy_score(y, p),
        "three_class_report": classification_report(y, p, labels=VERDICTS, zero_division=0, output_dict=True),
        "gold_distribution": dict(Counter(y)), "prediction_distribution": dict(Counter(p)),
        "localization": {
            "n_attack_success_with_gold_components": len(localized),
            "component_micro_precision": precision, "component_micro_recall": recall, "component_micro_f1": f1,
            "component_hit_rate": sum(bool(set(r["gold_components"]) & set(r["pred_components"])) for r in localized) / len(localized),
            "component_exact_match": sum(set(r["gold_components"]) == set(r["pred_components"]) for r in localized) / len(localized),
            "scope_accuracy": sum(r["gold_scope"] == r["pred_scope"] for r in localized) / len(localized),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--audit-adapter", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-batch", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    args.output_dir.mkdir(parents=True, exist_ok=True); args.cache_dir.mkdir(parents=True, exist_ok=True)
    caches = {}
    for split, path in (("train", args.train_file), ("validation", args.validation_file)):
        cache = args.cache_dir / f"{split}.pt"; caches[split] = cache
        if not cache.is_file():
            build_cache(path, cache, args.model, args.revision, args.audit_adapter, args.max_len, args.candidate_batch)
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["data_sha256"] != sha256(path):
            raise RuntimeError(f"Stale {split} feature cache")
        adapter_manifest = Path(args.audit_adapter) / "run_manifest.json"
        if not adapter_manifest.is_file():
            raise RuntimeError("Frozen V20 audit adapter has no run_manifest.json")
        if payload["adapter_manifest_sha256"] != sha256(adapter_manifest):
            raise RuntimeError(f"Stale {split} feature cache: adapter contract changed")
    train = FeatureDataset(caches["train"]); validation = FeatureDataset(caches["validation"])
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation, batch_size=args.batch, shuffle=False, collate_fn=collate)
    device = torch.device("cuda")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    model = Heads(train.payload["hidden_size"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    verdict_counts = Counter(row["verdict"] for row in train.rows)
    verdict_weights = torch.tensor([
        math.sqrt(len(train) / (len(VERDICTS) * verdict_counts[index])) for index in range(3)
    ], device=device)
    positives = sum(float(row["candidate_labels"].sum()) for row in train.rows)
    total = sum(len(row["candidate_labels"]) for row in train.rows)
    pos_weight = torch.tensor((total - positives) / max(positives, 1.0), device=device)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"v21_heads_epoch_{epoch}"):
            batch = move(batch, device); optimizer.zero_grad(set_to_none=True)
            verdict, scope, component = model(batch)
            loss = (
                F.cross_entropy(verdict, batch["verdict"], weight=verdict_weights)
                + F.cross_entropy(scope, batch["scope"])
                + F.binary_cross_entropy_with_logits(component, batch["candidate_labels"], pos_weight=pos_weight)
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_rows = records(model, validation_loader, device)
        cutoff = threshold(validation_rows); report = metrics(validation_rows, cutoff)
        score = report["three_class_report"]["macro avg"]["f1-score"] + report["localization"]["component_micro_f1"]
        if best is None or score > best[0]:
            best = (score, epoch, cutoff, report)
            torch.save(model.state_dict(), args.output_dir / "heads.pt")
    model.load_state_dict(torch.load(args.output_dir / "heads.pt", map_location=device, weights_only=True))
    validation_rows = records(model, validation_loader, device)
    cutoff = best[2]; report = metrics(validation_rows, cutoff)
    with (args.output_dir / "validation_controls.jsonl").open("w", encoding="utf-8") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.update({
        "model": args.model, "model_revision": args.revision,
        "audit_adapter": args.audit_adapter, "best_epoch": best[1],
        "frozen_qwen": True, "classification_lora": False, "localization_lora": False,
        "train_sha256": sha256(args.train_file), "validation_sha256": sha256(args.validation_file),
        "dataset_role": "validation", "sealed_test_accessed": False,
    })
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({
        "heads_sha256": sha256(args.output_dir / "heads.pt"),
        "controls_sha256": sha256(args.output_dir / "validation_controls.jsonl"),
        "train_sha256": sha256(args.train_file),
        "validation_sha256": sha256(args.validation_file),
        "audit_adapter_manifest_sha256": sha256(Path(args.audit_adapter) / "run_manifest.json"),
        "frozen_qwen": True,
        "sealed_test_accessed": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
