import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "v22_plain_hetero_agent.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v22_plain_hetero_agent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def prediction(run_id, guess, gold, *, refs=4):
    report = {"decision": {"confidence": "high"}, "audit_trace": [{}, {}]}
    return {
        "run_id": run_id, "pred": guess, "gold": gold, "pred_scope": "global",
        "pred_components": ["G::run"] if guess == "attack_success" else [],
        "pred_surface": "input", "pred_objective": "disruption",
        "generation": json.dumps(report),
        "trace_quality": {"valid_json": True, "evidence_refs": refs, "valid_evidence_refs": refs},
    }


def test_qwen_features_are_gold_blind():
    row = prediction("r", "attack_success", "clean_safe")
    first = MODULE.qwen_features(row, "track")
    row["gold"] = "attack_success"
    assert MODULE.qwen_features(row, "track") == first


def test_selector_features_are_gold_blind():
    q = prediction("r", "attack_failed", "attack_success")
    b = prediction("r", "attack_success", "attack_success")
    first = MODULE.selector_features(q, b, "track")
    q["gold"], b["gold"] = "clean_safe", "clean_safe"
    assert MODULE.selector_features(q, b, "track") == first


def test_learned_policy_trains_with_fixed_disjoint_split():
    q_rows, b_rows, index = [], [], []
    verdicts = ("clean_safe", "attack_failed", "attack_success")
    for i in range(120):
        run_id, gold = f"r{i}", verdicts[i % 3]
        q_guess = gold if i % 4 else verdicts[(i + 1) % 3]
        b_guess = gold if i % 5 else verdicts[(i + 2) % 3]
        q_rows.append(prediction(run_id, q_guess, gold, refs=1 + i % 6))
        b_rows.append(prediction(run_id, b_guess, gold))
        index.append({"run_id": run_id, "track": f"t{i % 2}"})
    policy, router, _ = MODULE.fit_learned_policy(q_rows, b_rows, index, 42)
    assert policy["split"]["router_train_rows"] + policy["split"]["calibration_rows"] == 120
    assert policy["router_target"] == "plain_qwen_prediction_is_wrong"
    assert len(router[0].get_feature_names_out()) == len(policy["router_model"]["coefficients"])
