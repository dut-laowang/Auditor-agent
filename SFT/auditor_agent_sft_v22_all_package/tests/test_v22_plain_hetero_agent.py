import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "v22_plain_hetero_agent.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v22_plain_hetero_agent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def pred(run_id, guess, gold):
    return {"run_id": run_id, "pred": guess, "gold": gold}


def test_calibration_selects_reliable_verifier_on_supported_conflicts():
    q, b, index = [], [], []
    for i in range(20):
        run_id = f"r{i}"
        q.append(pred(run_id, "attack_failed", "attack_success"))
        b.append(pred(run_id, "attack_success", "attack_success"))
        index.append({"run_id": run_id, "track": "track"})
    result = MODULE.calibrate(q, b, index, min_support=20)
    assert result["overall"]["choice"] == "bert"
    assert result["tracks"]["track"]["choice"] == "bert"


def test_calibration_defers_when_models_are_tied():
    q, b, index = [], [], []
    for i in range(20):
        run_id = f"r{i}"
        gold = "attack_success" if i % 2 else "attack_failed"
        q.append(pred(run_id, "attack_failed", gold))
        b.append(pred(run_id, "attack_success", gold))
        index.append({"run_id": run_id, "track": "track"})
    assert MODULE.calibrate(q, b, index, 20)["overall"]["choice"] == "defer"


def test_risk_priority_never_uses_gold():
    row = {"pred": "attack_success", "gold": "clean_safe", "trace_quality": {
        "valid_json": True, "evidence_refs": 4, "valid_evidence_refs": 4}}
    first = MODULE.risk_priority(row)
    row["gold"] = "attack_success"
    assert MODULE.risk_priority(row) == first
