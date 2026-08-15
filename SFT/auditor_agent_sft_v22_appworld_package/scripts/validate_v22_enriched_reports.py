from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = {"causal_explanation", "recommended_action", "confidence"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predictions = read(args.predictions)
    validation = read(args.validation)
    if len(predictions) != 406 or len(validation) != 406:
        raise RuntimeError("Enhanced V22 evaluation requires exactly 406 frozen rows")
    by_id = {row["metadata"]["run_id"]: row for row in validation}
    valid_json = fields_complete = refs_valid = 0
    for item in predictions:
        run_id = item["run_id"]
        if run_id not in by_id:
            raise RuntimeError(f"Unknown validation run_id: {run_id}")
        try:
            report = json.loads(item["generation"])
        except (TypeError, json.JSONDecodeError):
            continue
        valid_json += 1
        decision = report.get("decision", {})
        if not FIELDS.issubset(decision) or decision.get("confidence") not in {"high", "medium", "low"}:
            continue
        if not all(isinstance(decision[key], str) and decision[key].strip() for key in FIELDS):
            continue
        fields_complete += 1
        source = by_id[run_id]
        user = json.loads(source["messages"][1]["content"])
        evidence = user.get("run_evidence", {})
        available = {
            str(event["id"])
            for event in evidence.get("observed", []) + evidence.get("final_output", [])
            if isinstance(event, dict) and event.get("id")
        }
        cited = set(re.findall(r"\b(?:obs|out)_\d+\b", decision["causal_explanation"]))
        if cited and cited.issubset(available):
            refs_valid += 1
    result = {
        "rows": 406,
        "valid_json_rate": valid_json / 406,
        "three_enriched_fields_rate": fields_complete / 406,
        "causal_evidence_refs_valid_rate": refs_valid / 406,
        "validation_gold_used_as_model_input": False,
        "quality_gate": "PASS" if valid_json == fields_complete == refs_valid == 406 else "FAIL",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
