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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen ModernBERT 8192-token eligibility rule to final test.")
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--track-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    test_file = args.test_file.resolve()
    track_index = args.track_index.resolve()
    output = args.output_dir.resolve()
    manifest_path = output / "MODERNBERT_TEST_CONTEXT_GATE.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = (
            manifest.get("status") == "PASS",
            manifest.get("source_sha256") == sha256(test_file),
            manifest.get("source_track_index_sha256") == sha256(track_index),
            manifest.get("eligible_sha256") == sha256(output / "test.jsonl"),
            manifest.get("eligible_track_index_sha256") == sha256(output / "track_index.jsonl"),
        )
        if not all(checks):
            raise RuntimeError("Existing ModernBERT test context gate does not match its manifest")
        print(json.dumps({"reused": True, **manifest}, ensure_ascii=False, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing incomplete/non-empty context-gate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(test_file)
    index = read_jsonl(track_index)
    if [row["metadata"]["run_id"] for row in rows] != [row["run_id"] for row in index]:
        raise RuntimeError("Test and track-index run_id order mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    eligible: list[tuple[dict, dict]] = []
    excluded: list[dict] = []
    for position, (row, index_row) in enumerate(zip(rows, index)):
        user = next(message["content"] for message in row["messages"] if message.get("role") == "user")
        tokens = len(tokenizer(user, truncation=False, add_special_tokens=True)["input_ids"])
        if tokens <= args.max_len:
            eligible.append((row, index_row))
        else:
            excluded.append({
                "position": position,
                "run_id": index_row["run_id"],
                "track": index_row["track"],
                "verdict": index_row["verdict"],
                "tokens": tokens,
            })
    with (output / "test.jsonl").open("w", encoding="utf-8", newline="\n") as data_out, \
         (output / "track_index.jsonl").open("w", encoding="utf-8", newline="\n") as index_out:
        for row, index_row in eligible:
            data_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            index_out.write(json.dumps(index_row, ensure_ascii=False) + "\n")
    with (output / "excluded_overlength.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in excluded:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    track_source = {track: sum(item["track"] == track for item in index) for track in sorted({item["track"] for item in index})}
    track_eligible = {track: sum(item["track"] == track for _, item in eligible) for track in track_source}
    manifest = {
        "version": "V22-ALL-ModernBERT-test-context-gate-v1",
        "status": "PASS",
        "policy": "exclude_user_documents_over_8192_tokens_without_truncation",
        "model": args.model,
        "revision": args.revision,
        "max_len": args.max_len,
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(excluded),
        "coverage": len(eligible) / len(rows),
        "source_tracks": track_source,
        "eligible_tracks": track_eligible,
        "source_sha256": sha256(test_file),
        "source_track_index_sha256": sha256(track_index),
        "eligible_sha256": sha256(output / "test.jsonl"),
        "eligible_track_index_sha256": sha256(output / "track_index.jsonl"),
        "excluded_sha256": sha256(output / "excluded_overlength.jsonl"),
        "eligibility_uses_labels": False,
        "truncation_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
