from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic exclusion set for the strict V23 release."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--context-exclusions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=("minimal", "strict"), default="strict")
    args = parser.parse_args()

    context_payload = json.loads(args.context_exclusions.read_text(encoding="utf-8"))
    context_ids = {str(value) for value in context_payload.get("run_ids", [])}
    reasons: dict[str, set[str]] = defaultdict(set)
    pools: dict[str, str] = {}

    for run_id in context_ids:
        reasons[run_id].add("qwen_context_overflow")

    for all_path in sorted(args.candidates.glob("*/all.jsonl")):
        pool = all_path.parent.name
        for raw_line, row in iter_jsonl(all_path):
            metadata = row["metadata"]
            run_id = str(metadata["run_id"])
            pools[run_id] = pool
            source_label = metadata.get("source_final_label")
            consensus = metadata.get("semantic_consensus")
            if args.policy == "strict" and metadata.get("label_quality") == "silver":
                reasons[run_id].add("silver_label")
            if args.policy == "strict" and (
                consensus
                and source_label in {"success", "failure"}
                and consensus != source_label
            ):
                reasons[run_id].add("semantic_consensus_conflict")
            if "\ufffd" in raw_line:
                reasons[run_id].add("unicode_replacement_character")

    reason_counts = Counter()
    pool_counts = Counter()
    for run_id, run_reasons in reasons.items():
        for reason in run_reasons:
            reason_counts[reason] += 1
        pool_counts[pools.get(run_id, "preexisting_context_gate")] += 1

    payload = {
        "version": f"V23-{args.policy}-exclusions-v1",
        "policy": {
            "parent": "preserve every frozen V22 row unchanged",
            "new_rows": (
                [
                    "exclude Qwen examples exceeding the pinned full-context limit",
                    "exclude every row containing the Unicode replacement character",
                ]
                if args.policy == "minimal"
                else [
                    "exclude Qwen examples exceeding the pinned full-context limit",
                    "exclude every silver label",
                    "exclude every source-label/semantic-consensus conflict",
                    "exclude every row containing the Unicode replacement character",
                ]
            ),
        },
        "unique_excluded_run_ids": len(reasons),
        "reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "pool_counts_exclusive": dict(sorted(pool_counts.items())),
        "run_ids": sorted(reasons),
        "reasons_by_run_id": {
            run_id: sorted(run_reasons) for run_id, run_reasons in sorted(reasons.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"run_ids", "reasons_by_run_id"}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
