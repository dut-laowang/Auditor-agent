from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def render(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all V23 splits against the frozen Qwen context budget.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-len", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    report = {
        "version": "V23-Qwen-context-preflight-v1",
        "policy": "no truncation; reject only over-budget V23 additions; frozen V22 rows must remain eligible",
        "model": args.model,
        "revision": args.revision,
        "max_len": args.max_len,
        "splits": {},
        "status": "PASS",
    }
    exclusions = []
    old_over_budget = []
    for split in ("train", "validation", "test"):
        rows = read(args.data_dir / f"{split}.jsonl")
        index = read(args.data_dir / f"{split}_track_index.jsonl")
        if [row["metadata"]["run_id"] for row in rows] != [row["run_id"] for row in index]:
            raise RuntimeError(f"{split}: data/index order mismatch")
        over = []
        max_observed = 0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [render(tokenizer, row["messages"]) for row in batch]
            lengths = tokenizer(texts, add_special_tokens=False, return_length=True)["length"]
            for offset, (row, idx, length) in enumerate(zip(batch, index[start : start + len(batch)], lengths)):
                max_observed = max(max_observed, int(length))
                if length <= args.max_len:
                    continue
                item = {
                    "position": start + offset,
                    "run_id": str(idx["run_id"]),
                    "track": idx["track"],
                    "split": split,
                    "verdict": idx["verdict"],
                    "tokens": int(length),
                    "dataset_version": row.get("metadata", {}).get("dataset_version", "V22"),
                }
                over.append(item)
                if item["dataset_version"] == "V23":
                    exclusions.append(item)
                else:
                    old_over_budget.append(item)
        report["splits"][split] = {
            "rows": len(rows),
            "over_budget_rows": len(over),
            "over_budget_by_track": dict(Counter(item["track"] for item in over)),
            "max_observed_tokens": max_observed,
            "over_budget": over,
        }
    if old_over_budget:
        report["status"] = "FAIL_FROZEN_V22_OVER_BUDGET"
    report["v23_exclusion_rows"] = len(exclusions)
    report["frozen_v22_over_budget_rows"] = len(old_over_budget)
    (args.output_dir / "V23_QWEN_CONTEXT_PREFLIGHT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "V23_CONTEXT_EXCLUSIONS.json").write_text(
        json.dumps(
            {
                "version": "V23-Qwen-context-exclusions-v1",
                "policy": report["policy"],
                "run_ids": [item["run_id"] for item in exclusions],
                "rows": exclusions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if old_over_budget:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
