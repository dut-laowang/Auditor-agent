from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = load(args.track_dir / "validation.jsonl")
    test_ids = {row["metadata"]["sample_uid"] for row in load(args.track_dir / "test.jsonl")}
    result = {"status": "PASS", "test_accessed_for_generation": False, "variants": {}}
    for path in sorted((args.track_dir / "validation_counterfactuals").glob("*.jsonl")):
        rows = load(path)
        same_order = len(rows) == len(base) and all(
            row["metadata"]["sample_uid"] == original["metadata"]["sample_uid"]
            for row, original in zip(rows, base)
        )
        targets_unchanged = len(rows) == len(base) and all(
            row["messages"][2]["content"] == original["messages"][2]["content"]
            for row, original in zip(rows, base)
        )
        changed_inputs = sum(
            row["messages"][1]["content"] != original["messages"][1]["content"]
            for row, original in zip(rows, base)
        )
        test_id_hits = sum(row["metadata"]["sample_uid"] in test_ids for row in rows)
        result["variants"][path.stem] = {
            "rows": len(rows),
            "same_validation_uid_order": same_order,
            "assistant_targets_unchanged": targets_unchanged,
            "changed_user_inputs": changed_inputs,
            "sealed_test_uid_hits": test_id_hits,
        }
        if not same_order or not targets_unchanged or not changed_inputs or test_id_hits:
            result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
