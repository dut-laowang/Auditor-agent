from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    report = {
        "policy": "drop_complete_chat_sequences_exceeding_qwen_context",
        "model": args.model,
        "revision": args.revision,
        "max_len": args.max_len,
        "splits": {},
    }
    for split in ("train", "validation"):
        source = args.input_dir / f"{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        kept = []
        dropped = []
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                length = len(
                    tokenizer(
                        render(tokenizer, row["messages"]), add_special_tokens=False
                    )["input_ids"]
                )
                if length <= args.max_len:
                    kept.append(row)
                else:
                    dropped.append(
                        {
                            "index": index,
                            "sample_uid": row.get("metadata", {}).get("sample_uid"),
                            "tokens": length,
                        }
                    )
        with destination.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "source_rows": len(kept) + len(dropped),
            "kept_rows": len(kept),
            "dropped_rows": len(dropped),
            "dropped": dropped,
            "source_sha256": sha256(source),
            "filtered_sha256": sha256(destination),
        }
    (args.output_dir / "context_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
