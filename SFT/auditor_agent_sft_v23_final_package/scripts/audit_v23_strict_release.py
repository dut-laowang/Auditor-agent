from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")
SPLITS = ("train", "validation", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify strict-only V23 release invariants.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=("minimal", "strict"), default="strict")
    args = parser.parse_args()

    exclusion_payload = json.loads(args.exclusions.read_text(encoding="utf-8"))
    excluded_ids = set(map(str, exclusion_payload["run_ids"]))
    pending_queue_ids: set[str] = set()
    for queue_path in args.candidate_root.glob("*/manual_review_queue*.json"):
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        for item in queue.get("items", []):
            pending_queue_ids.add(str(item["sample"]["metadata"]["run_id"]))

    errors: list[str] = []
    counts = Counter()
    retained_ids: set[str] = set()
    new_ids: set[str] = set()
    retained_quality = Counter()
    for track in TRACKS:
        for split in SPLITS:
            path = args.data / track / f"{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    metadata = row["metadata"]
                    run_id = str(metadata["run_id"])
                    retained_ids.add(run_id)
                    counts[f"split::{split}"] += 1
                    counts[f"track::{track}"] += 1
                    counts[f"verdict::{metadata['verdict']}"] += 1
                    if not metadata.get("dataset_version"):
                        counts["origin::V22"] += 1
                        continue
                    new_ids.add(run_id)
                    counts["origin::V23_increment"] += 1
                    source_label = metadata.get("source_final_label")
                    consensus = metadata.get("semantic_consensus")
                    if metadata.get("label_quality") == "silver":
                        retained_quality["silver"] += 1
                    if consensus and source_label in {"success", "failure"} and consensus != source_label:
                        retained_quality["semantic_conflict"] += 1
                    if "\ufffd" in line:
                        retained_quality["replacement_character"] += 1
                    if args.policy == "strict" and metadata.get("label_quality") == "silver":
                        errors.append(f"silver label retained: {track}/{split}:{line_number}")
                    if (
                        args.policy == "strict"
                        and consensus
                        and source_label in {"success", "failure"}
                        and consensus != source_label
                    ):
                        errors.append(f"semantic conflict retained: {track}/{split}:{line_number}")
                    if "\ufffd" in line:
                        errors.append(f"replacement character retained: {track}/{split}:{line_number}")

    # An excluded candidate run ID may legitimately already exist in frozen V22.
    # The strict exclusion policy applies only to appended V23 rows.
    excluded_retained = excluded_ids & new_ids
    queue_retained = pending_queue_ids & new_ids
    if excluded_retained:
        errors.append(f"strict exclusion IDs retained: {len(excluded_retained)}")
    if args.policy == "strict" and queue_retained:
        errors.append(f"pending review queue IDs retained: {len(queue_retained)}")

    report = {
        "version": f"V23-{args.policy}-release-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "rows_checked": len(retained_ids),
        "new_rows_checked": len(new_ids),
        "counts": dict(sorted(counts.items())),
        "strict_exclusion_ids_retained": len(excluded_retained),
        "pending_review_queue_ids_retained": len(queue_retained),
        "new_silver_labels_retained": retained_quality["silver"],
        "new_semantic_conflicts_retained": retained_quality["semantic_conflict"],
        "new_replacement_character_rows_retained": retained_quality["replacement_character"],
        "errors": errors[:100],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
