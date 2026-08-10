"""Deterministic smoke tests for V19 target parsing and metric logic."""

import json
import math
import sys
from pathlib import Path

SERVER_SCRIPTS = Path(__file__).resolve().parents[1] / "server_scripts"
sys.path.insert(0, str(SERVER_SCRIPTS))

import eval_qwen3_fullschema_v19 as qwen_eval
import modernbert_multitask_v19 as modernbert


def assert_close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"{name}: {actual} != {expected}")


def main():
    target = {
        "decision": {"verdict": "attack_success", "binary_label": "unsafe"},
        "localization": {"scope": "node", "component_ids": ["N::a"]},
    }
    rendered = "prefix\n" + json.dumps(target) + "\nsuffix"
    if qwen_eval.extract_verdict(rendered) != "attack_success":
        raise AssertionError("Qwen verdict parser failed")
    if qwen_eval.extract_localization(rendered) != ("node", {"N::a"}):
        raise AssertionError("Qwen localization parser failed")
    records = [
        {
            "run_id": "r1",
            "gold": "attack_success",
            "pred": "attack_success",
            "gold_scope": "node",
            "pred_scope": "node",
            "gold_components": ["a", "b"],
            "pred_components": [],
            "component_ids": ["a", "b", "x"],
            "component_probabilities": [0.9, 0.4, 0.1],
        },
        {
            "run_id": "r2",
            "gold": "attack_success",
            "pred": "attack_failed",
            "gold_scope": "edge",
            "pred_scope": "node",
            "gold_components": ["c"],
            "pred_components": [],
            "component_ids": ["c", "d"],
            "component_probabilities": [0.8, 0.7],
        },
        {
            "run_id": "r3",
            "gold": "clean_safe",
            "pred": "clean_safe",
            "gold_scope": "none",
            "pred_scope": "none",
            "gold_components": [],
            "pred_components": [],
            "component_ids": ["z"],
            "component_probabilities": [0.2],
        },
    ]
    metrics = modernbert.metrics_from_records(records, threshold=0.5)
    assert_close(metrics["three_class_accuracy"], 2 / 3, "three_class_accuracy")
    localization = metrics["localization"]
    assert_close(localization["component_micro_precision"], 2 / 3, "component_precision")
    assert_close(localization["component_micro_recall"], 2 / 3, "component_recall")
    assert_close(localization["component_micro_f1"], 2 / 3, "component_f1")
    assert_close(localization["component_hit_rate"], 1.0, "component_hit_rate")
    assert_close(localization["component_exact_match"], 0.0, "component_exact_match")
    assert_close(localization["scope_accuracy"], 0.5, "scope_accuracy")
    print(json.dumps({"status": "PASS", "tests": 9}, indent=2))


if __name__ == "__main__":
    main()
