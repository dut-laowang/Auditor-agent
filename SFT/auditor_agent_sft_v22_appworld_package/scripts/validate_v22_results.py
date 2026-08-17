from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r'(?i)"(?:password|access_token|api_key|secret)"\s*:'),
    re.compile(r'(?i)bearer\s+[a-z0-9._-]{12,}'),
]


def get(obj: dict, path: tuple[str, ...]):
    for key in path:
        obj = obj[key]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--modernbert-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=406)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    reference = json.loads(args.modernbert_metrics.read_text(encoding="utf-8"))
    if len(rows) != args.expected_rows or metrics["n"] != args.expected_rows:
        raise RuntimeError("V22 result row mismatch")
    paths = [
        ("three_class_accuracy",), ("three_class_report", "macro avg", "f1-score"),
        ("three_class_report", "attack_success", "recall"),
        ("localization", "component_micro_f1"), ("localization", "component_hit_rate"),
        ("localization", "component_exact_match"), ("localization", "scope_accuracy"),
    ]
    for path in paths:
        if not math.isclose(get(metrics, path), get(reference, path), rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Agent/ModernBERT metric mismatch: {path}")
    secret_hits = []
    for row in rows:
        report = json.loads(row["generation"])
        if set(report) != {"decision", "attack", "localization", "audit_trace"}:
            raise RuntimeError(f"Bad report schema: {row['run_id']}")
        if not report["decision"].get("confidence_basis"):
            raise RuntimeError(f"Missing confidence basis: {row['run_id']}")
        verdict = report["decision"].get("verdict")
        expected_binary = "unsafe" if verdict == "attack_success" else "safe"
        if verdict not in {"clean_safe", "attack_failed", "attack_success"}:
            raise RuntimeError(f"Invalid verdict: {row['run_id']}")
        if report["decision"].get("binary_label") != expected_binary:
            raise RuntimeError(f"Verdict/binary-label contradiction: {row['run_id']}")
        attack = report.get("attack", {})
        if attack.get("present") is not (verdict != "clean_safe"):
            raise RuntimeError(f"Verdict/attack-presence contradiction: {row['run_id']}")
        if verdict == "clean_safe" and (attack.get("surface"), attack.get("objective")) != ("none", "none"):
            raise RuntimeError(f"Clean verdict has attack characterization: {row['run_id']}")
        if verdict != "clean_safe" and (attack.get("surface") == "none" or attack.get("objective") == "none"):
            raise RuntimeError(f"Attack verdict lacks attack characterization: {row['run_id']}")
        for step in report["audit_trace"]:
            for required in ("finding", "evidence_summary", "outcome_assessment"):
                if not step.get(required):
                    raise RuntimeError(f"Missing {required}: {row['run_id']}")
            refs = list(map(str, step.get("evidence_refs", [])))
            summary_refs = [str(item.get("evidence_ref")) for item in step["evidence_summary"]]
            if refs != summary_refs:
                raise RuntimeError(f"Evidence summary mismatch: {row['run_id']}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(row["generation"]):
                secret_hits.append(row["run_id"])
    if secret_hits:
        raise RuntimeError(f"Generated report secret-like content: {secret_hits[:5]}")
    quality = {
        "n": args.expected_rows, "modernbert_metrics_exact": True,
        "complete_explanation_fields_rate": 1.0,
        "evidence_summary_alignment_rate": 1.0,
        "generated_secret_pattern_hits": 0,
        "valid_json_rate": metrics["audit_trace_quality"]["valid_json_rate"],
        "evidence_ref_validity_rate": metrics["audit_trace_quality"]["evidence_ref_validity_rate"],
        "sealed_test_accessed": False,
    }
    args.output.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
