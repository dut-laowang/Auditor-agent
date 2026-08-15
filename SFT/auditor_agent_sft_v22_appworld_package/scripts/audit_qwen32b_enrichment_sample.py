from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v22-train", required=True, type=Path)
    parser.add_argument("--teacher-output", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()
    source = read(args.v22_train)[: args.expected]
    teacher = read(args.teacher_output)
    if len(source) != args.expected or len(teacher) != args.expected:
        raise RuntimeError(f"Small-batch row mismatch: {len(source)} != {len(teacher)}")
    confidence = {"high": 0, "medium": 0, "low": 0}
    for index, (row, item) in enumerate(zip(source, teacher)):
        run_id = row["metadata"]["run_id"]
        if item.get("run_id") != run_id:
            raise RuntimeError(f"Run-id mismatch at {index}")
        extra = item.get("enrichment", {})
        if set(extra) != {"causal_explanation", "recommended_action", "confidence"}:
            raise RuntimeError(f"Wrong fields at {run_id}")
        confidence[extra["confidence"]] += 1
        user = json.loads(row["messages"][1]["content"])
        evidence = user.get("run_evidence", {})
        observable = {
            str(event["id"])
            for event in evidence.get("observed", []) + evidence.get("final_output", [])
            if isinstance(event, dict) and event.get("id")
        }
        report = json.loads(row["messages"][2]["content"])
        available = {
            str(ref)
            for step in report.get("audit_trace", [])
            for ref in step.get("evidence_refs", [])
        }
        if not available.issubset(observable):
            raise RuntimeError(f"Gold audit trace contains invisible evidence at {run_id}")
        cited = set(re.findall(r"\b(?:obs|out)_\d+\b", extra["causal_explanation"]))
        required = min(2, len(available))
        if len(cited) < required or not cited.issubset(available):
            raise RuntimeError(f"Invalid causal evidence at {run_id}")
    result = {
        "quality_gate": "PASS",
        "rows": len(teacher),
        "run_ids_ordered_and_preserved": True,
        "only_three_enrichment_fields": True,
        "causal_evidence_refs_valid": True,
        "confidence_distribution": confidence,
        "source_train_modified": False,
    }
    output = args.teacher_output.with_name("SMALL_BATCH_QUALITY.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
