from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen user-only ModernBERT context gate for expanded V22.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    report = {
        "version": "V22-ALL-expanded-ModernBERT-user-context-v1",
        "status": "PASS",
        "input_mode": "user",
        "model": args.model,
        "revision": args.revision,
        "max_len": args.max_len,
        "splits": {},
    }
    for split in args.splits:
        data_path = args.data_dir / f"{split}.jsonl"
        index_path = args.index_dir / ("track_index.jsonl" if split == "test" else f"{split}.jsonl")
        rows, index = read(data_path), read(index_path)
        if [row["metadata"]["run_id"] for row in rows] != [row["run_id"] for row in index]:
            raise RuntimeError(f"{split}: data/index run_id order mismatch")
        kept_rows, kept_index, excluded = [], [], []
        for position, (row, idx) in enumerate(zip(rows, index)):
            user = next(item["content"] for item in row["messages"] if item.get("role") == "user")
            length = len(tokenizer(user, truncation=False, add_special_tokens=True)["input_ids"])
            if length <= args.max_len:
                kept_rows.append(row)
                kept_index.append(idx)
            else:
                excluded.append({
                    "position": position, "run_id": idx["run_id"], "track": idx["track"],
                    "verdict": idx["verdict"], "tokens": length,
                })
        out_data = args.output_dir / f"{split}.jsonl"
        out_index = args.output_dir / f"{split}_track_index.jsonl"
        write(out_data, kept_rows)
        write(out_index, kept_index)
        report["splits"][split] = {
            "source_rows": len(rows), "eligible_rows": len(kept_rows),
            "excluded_rows": len(excluded),
            "excluded_by_track": dict(Counter(item["track"] for item in excluded)),
            "excluded": excluded,
            "source_sha256": sha256(data_path), "source_index_sha256": sha256(index_path),
            "eligible_sha256": sha256(out_data), "eligible_index_sha256": sha256(out_index),
        }
    (args.output_dir / "MODERNBERT_CONTEXT_GATE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
