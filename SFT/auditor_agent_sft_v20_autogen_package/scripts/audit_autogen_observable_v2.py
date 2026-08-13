from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


PRIVILEGED_KEYS = (
    "attack_exposed", "attack_injected", "attack_active", "attack_payload_template",
    "source_final_label", "marker_success", "success_markers", "security_judgment",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = {name: read_jsonl(args.data_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    successes = [row for rows in splits.values() for row in rows if row["metadata"]["verdict"] == "attack_success"]
    rng = random.Random(20260813)
    by_surface: dict[str, list[dict]] = {}
    for row in successes:
        by_surface.setdefault(row["metadata"]["surface"], []).append(row)
    sample: list[dict] = []
    for surface in sorted(by_surface):
        pool = by_surface[surface]
        sample.extend(rng.sample(pool, min(5, len(pool))))
    remaining = [row for row in successes if row not in sample]
    sample.extend(rng.sample(remaining, min(50 - len(sample), len(remaining))))
    problems = Counter()
    for row in [item for rows in splits.values() for item in rows]:
        user = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        visible = json.dumps(user, ensure_ascii=False).casefold()
        for key in PRIVILEGED_KEYS:
            problems[f"privileged::{key}"] += key in visible
        evidence = {
            event["id"]: event
            for section in ("observed", "final_output")
            for event in user["run_evidence"].get(section, [])
        }
        candidates = {item["id"] for item in user["graph_candidates"]}
        problems["missing_gold_candidate"] += any(
            component not in candidates
            for component in answer["localization"]["component_ids"]
        )
        problems["invalid_evidence_ref"] += any(
            ref not in evidence
            for trace in answer["audit_trace"]
            for ref in trace["evidence_refs"]
        )
        problems["empty_final_output"] += not user["run_evidence"].get("final_output")
        problems["completed_only_final_output"] += bool(user["run_evidence"].get("final_output")) and all(
            str(event.get("text") or "").strip().casefold() == "completed"
            for event in user["run_evidence"]["final_output"]
        )
    sample_checks = Counter()
    sample_surfaces = Counter()
    for row in sample:
        user = json.loads(row["messages"][1]["content"])
        sample_surfaces[row["metadata"]["surface"]] += 1
        sample_checks["substantive_final_output"] += any(
            len(str(event.get("text") or "").strip()) > 20
            for event in user["run_evidence"]["final_output"]
        )
    report = {
        "status": "PASS" if not any(problems.values()) and sample_checks["substantive_final_output"] == len(sample) else "FAIL",
        "policy": "original AutoGen delivered_content only; no attack instrumentation or config payload is visible",
        "splits": {name: len(rows) for name, rows in splits.items()},
        "attack_success_rows": len(successes),
        "problems": dict(problems),
        "stratified_quality_sample": {
            "seed": 20260813,
            "n": len(sample),
            "surface_distribution": dict(sample_surfaces),
            "substantive_final_output": sample_checks["substantive_final_output"],
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
