from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


FORBIDDEN_KEYS = {
    "attack_id", "attack_metadata", "success_marker", "success_markers",
    "source_final_label", "semantic_consensus", "label_quality", "gold_label",
    "judge_conclusion", "marker_success", "payload_id",
}
FORBIDDEN_TEXT = re.compile(
    r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|success_markers?|attack_metadata|"
    r"source_final_label|semantic_consensus|judge_conclusion|gold_label",
    re.I,
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = {name: load(args.data_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    report = {"status": "PASS", "test_model_evaluation_performed": False, "splits": {k: len(v) for k, v in splits.items()}}

    sets = {}
    for name, rows in splits.items():
        sets[name] = {
            "source_run": {(r["metadata"]["source_type"], r["metadata"]["run_id"]) for r in rows},
            "task": {(r["metadata"]["scenario"], int(r["metadata"]["sample_id"])) for r in rows},
            "input": {hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest() for r in rows},
            "normalized_input": {hashlib.sha256(normalize(r["messages"][1]["content"]).encode()).hexdigest() for r in rows},
        }
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}"] = {key: len(sets[left][key] & sets[right][key]) for key in sets[left]}
    report["overlaps"] = overlaps

    problems = Counter()
    dual = Counter()
    for split, rows in splits.items():
        for row in rows:
            user_text = row["messages"][1]["content"]
            user = json.loads(user_text)
            assistant = json.loads(row["messages"][2]["content"])
            if FORBIDDEN_TEXT.search(user_text):
                problems[f"{split}:forbidden_literal"] += 1
            if set(keys(user)) & FORBIDDEN_KEYS:
                problems[f"{split}:forbidden_key"] += 1
            evidence_ids = {
                event["id"]
                for event in user["run_evidence"].get("observed", []) + user["run_evidence"].get("final_output", [])
                if event.get("id")
            }
            candidates = {candidate["id"]: candidate for candidate in user.get("graph_candidates", [])}
            component_ids = assistant["localization"].get("component_ids", [])
            if any(component not in candidates for component in component_ids):
                problems[f"{split}:invalid_component_ref"] += 1
            for step in assistant.get("audit_trace", []):
                if any(ref not in evidence_ids for ref in step.get("evidence_refs", [])):
                    problems[f"{split}:invalid_evidence_ref"] += 1
            if row["metadata"].get("attack_mode") == "dual_site" and row["metadata"].get("verdict") != "clean_safe":
                dual[(split, assistant["localization"].get("scope"), len(component_ids))] += 1

    report["problems"] = dict(problems)
    report["dual_projection"] = {"|".join(map(str, key)): value for key, value in sorted(dual.items())}
    report["sealed_test_consumption_record_present"] = any(args.data_dir.rglob("SEALED_TEST_CONSUMED.json"))
    if problems or report["sealed_test_consumption_record_present"] or any(
        value for pair in overlaps.values() for value in pair.values()
    ):
        report["status"] = "FAIL"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
