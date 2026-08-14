from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


FORBIDDEN = {
    "attack_exposed", "attack_injected", "attack_active", "attacked_content",
    "attack_payload_template", "source_final_label", "marker_success",
    "success_markers", "security_judgment", "appworld_oracle",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for split in ("train", "validation", "test"):
        with (args.data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    problems = Counter()
    success_rows = []
    for row in rows:
        user = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        serialized = json.dumps(user, ensure_ascii=False).casefold()
        for key in FORBIDDEN:
            problems[f"privileged::{key}"] += f'"{key.casefold()}"' in serialized
        evidence = {item["id"] for section in ("observed", "final_output") for item in user["run_evidence"].get(section, [])}
        problems["invalid_evidence_ref"] += any(ref not in evidence for trace in answer["audit_trace"] for ref in trace["evidence_refs"])
        problems["empty_final_output"] += not user["run_evidence"].get("final_output")
        if row["metadata"]["verdict"] == "attack_success":
            success_rows.append(row)
    rng = random.Random(20260814)
    rng.shuffle(success_rows)
    sample = success_rows[:50]
    report = {
        "status": "PASS" if not any(problems.values()) else "FAIL",
        "policy": "framework-visible AppWorld×MARBLE fields only; no attack instrumentation or oracle fields in model input",
        "rows": len(rows), "attack_success_rows": len(success_rows),
        "problems": dict(problems),
        "quality_sample": {"seed": 20260814, "n": len(sample), "run_ids": [r["metadata"]["run_id"] for r in sample]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
