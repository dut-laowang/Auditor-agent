#!/usr/bin/env python3
"""Build leakage-audited held-out folds from the final V23 training corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SOURCE_COUNTS = {"train": 30619, "validation": 7018}
DEFAULT_FOLDS = (("topology", "tree"), ("surface", "message"), ("scenario", "research"))
MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def uid(row: dict) -> str:
    meta = row["metadata"]
    return str(meta.get("sample_uid") or meta["run_id"])


def group(row: dict) -> str:
    meta = row["metadata"]
    return str(meta.get("task_group_id") or f"{meta.get('source_type')}::{meta.get('scenario')}::{meta.get('sample_id')}" or meta["run_id"])


def exact_input(row: dict) -> str:
    visible = [m for m in row["messages"] if m["role"] != "assistant"]
    raw = json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def selected(row: dict, axis: str, value: str) -> bool:
    return str(row["metadata"].get(axis)) == value


def modernbert_eligible(rows: list[dict], tokenizer, max_len: int = 8192) -> tuple[list[dict], list[dict]]:
    """Apply the exact zero-truncation document/candidate contract used by the trainer."""
    kept, excluded = [], []
    for row in rows:
        user = row["messages"][1]["content"]
        document_len = len(tokenizer(user, truncation=False, add_special_tokens=True)["input_ids"])
        payload = json.loads(user)
        candidate_lengths = [
            len(tokenizer(json.dumps(c, ensure_ascii=False, sort_keys=True), truncation=False, add_special_tokens=True)["input_ids"])
            for c in payload.get("graph_candidates", []) if isinstance(c, dict) and c.get("id")
        ]
        candidate_max = max(candidate_lengths, default=0)
        if document_len <= max_len and candidate_max <= max_len:
            kept.append(row)
        else:
            excluded.append({"run_id": row["metadata"]["run_id"], "document_tokens": document_len, "max_candidate_tokens": candidate_max})
    return kept, excluded


def build_fold(train: list[dict], validation: list[dict], axis: str, value: str) -> tuple[list[dict], list[dict]]:
    if axis == "surface":
        # Clean controls have surface=none. Keep them in training and evaluation;
        # remove only the attacked examples of the held-out attack surface.
        fold_train = [r for r in train if not selected(r, axis, value)]
        fold_eval = [r for r in validation if selected(r, axis, value) or r["metadata"]["verdict"] == "clean_safe"]
    else:
        fold_train = [r for r in train if not selected(r, axis, value)]
        fold_eval = [r for r in validation if selected(r, axis, value)]
    return fold_train, fold_eval


def audit(train: list[dict], evaluation: list[dict], axis: str, value: str) -> dict:
    train_uids, eval_uids = {uid(r) for r in train}, {uid(r) for r in evaluation}
    train_groups, eval_groups = {group(r) for r in train}, {group(r) for r in evaluation}
    train_inputs, eval_inputs = {exact_input(r) for r in train}, {exact_input(r) for r in evaluation}
    if train_uids & eval_uids or train_groups & eval_groups or train_inputs & eval_inputs:
        raise RuntimeError(f"Leakage detected in held-out fold {axis}={value}")
    if any(selected(r, axis, value) for r in train):
        raise RuntimeError(f"Held-out value remains in training: {axis}={value}")
    if not evaluation:
        raise RuntimeError(f"Empty held-out evaluation: {axis}={value}")
    verdicts = collections.Counter(r["metadata"]["verdict"] for r in evaluation)
    if axis != "surface" and len(verdicts) < 2:
        raise RuntimeError(f"Degenerate held-out evaluation labels: {axis}={value}")
    return {
        "axis": axis,
        "value": value,
        "train_rows": len(train),
        "evaluation_rows": len(evaluation),
        "evaluation_verdicts": dict(sorted(verdicts.items())),
        "run_id_overlap": len(train_uids & eval_uids),
        "task_group_overlap": len(train_groups & eval_groups),
        "exact_visible_input_overlap": len(train_inputs & eval_inputs),
        "surface_protocol": "held-out attacked surface plus validation clean controls" if axis == "surface" else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold", action="append", help="axis=value; repeatable")
    p.add_argument("--modernbert-zero-truncation", action="store_true", help="Filter each fold using the pinned ModernBERT tokenizer and 8192-token hard gate")
    args = p.parse_args()
    train = read_jsonl(args.data_dir / "train.jsonl")
    validation = read_jsonl(args.data_dir / "validation.jsonl")
    if {"train": len(train), "validation": len(validation)} != SOURCE_COUNTS:
        raise RuntimeError("This suite accepts only final V23 data (30,619 train / 7,018 validation)")
    folds = DEFAULT_FOLDS if not args.fold else tuple(tuple(x.split("=", 1)) for x in args.fold)
    tokenizer = None
    if args.modernbert_zero_truncation:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL, revision=MODERNBERT_REVISION)
    manifest = {"version": "V23-final-heldout-v1", "source_counts": SOURCE_COUNTS, "folds": {}, "modernbert_zero_truncation": bool(tokenizer), "modernbert_revision": MODERNBERT_REVISION if tokenizer else None}
    for axis, value in folds:
        if axis not in {"topology", "surface", "scenario"}:
            raise ValueError(f"Unsupported held-out axis: {axis}")
        fold_train, fold_eval = build_fold(train, validation, axis, value)
        excluded_train, excluded_eval = [], []
        if tokenizer is not None:
            fold_train, excluded_train = modernbert_eligible(fold_train, tokenizer)
            fold_eval, excluded_eval = modernbert_eligible(fold_eval, tokenizer)
        report = audit(fold_train, fold_eval, axis, value)
        report["zero_truncation_excluded_train"] = len(excluded_train)
        report["zero_truncation_excluded_validation"] = len(excluded_eval)
        report["zero_truncation_exclusions"] = {"train": excluded_train, "validation": excluded_eval}
        root = args.output_dir / f"{axis}__{value}"
        report["train_sha256"] = write_jsonl(root / "train.jsonl", fold_train)
        report["validation_sha256"] = write_jsonl(root / "validation.jsonl", fold_eval)
        (root / "LEAKAGE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest["folds"][f"{axis}={value}"] = report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "HELDOUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
