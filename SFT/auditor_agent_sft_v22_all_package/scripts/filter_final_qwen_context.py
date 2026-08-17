from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def render(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact final-chat Qwen context gate for enriched V22-ALL data.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--track-index", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-len", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-validation-rows", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    tracks = {
        row["run_id"]: row["track"]
        for row in read(args.track_index)
    }
    report = {
        "version": "V22-ALL-final-qwen-context-v1",
        "policy": "preserve_all_rows; any_complete_chat_over_model_budget_fails_closed",
        "model": args.model,
        "revision": args.revision,
        "max_len": args.max_len,
        "splits": {},
        "status": "PASS",
    }
    for split in ("train", "validation"):
        source = args.input_dir / f"{split}.jsonl"
        kept, dropped = [], []
        rows = read(source)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [render(tokenizer, row["messages"]) for row in batch]
            lengths = tokenizer(texts, add_special_tokens=False, return_length=True)["length"]
            for offset, (row, length) in enumerate(zip(batch, lengths)):
                index = start + offset
                if length <= args.max_len:
                    kept.append(row)
                else:
                    run_id = str(row.get("metadata", {}).get("run_id") or "")
                    dropped.append({
                        "index": index,
                        "run_id": run_id,
                        "track": tracks.get(run_id, "unknown"),
                        "verdict": row.get("metadata", {}).get("verdict"),
                        "tokens": length,
                    })
        destination = args.output_dir / f"{split}.jsonl"
        if dropped:
            write(destination, kept)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        report["splits"][split] = {
            "source_rows": len(kept) + len(dropped),
            "kept_rows": len(kept),
            "dropped_rows": len(dropped),
            "dropped_by_track": dict(Counter(item["track"] for item in dropped)),
            "dropped": dropped,
            "source_sha256": sha256(source),
            "filtered_sha256": sha256(destination),
        }

    validation = report["splits"]["validation"]
    if (
        validation["source_rows"] != args.expected_validation_rows
        or validation["dropped_rows"]
        or report["splits"]["train"]["dropped_rows"]
    ):
        report["status"] = "FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "FINAL_QWEN_CONTEXT_GATE.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
