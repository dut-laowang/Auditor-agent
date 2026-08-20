import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "v22_agent_demo.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v22_agent_demo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def prediction(run_id, pred="clean_safe", *, valid=True, refs=2, valid_refs=2):
    return {
        "run_id": run_id,
        "gold": "attack_success",
        "pred": pred,
        "pred_binary": "unsafe" if pred == "attack_success" else "safe",
        "pred_scope": "global",
        "pred_components": ["G::run"] if pred == "attack_success" else [],
        "pred_surface": "input" if pred != "clean_safe" else "none",
        "trace_quality": {"valid_json": valid, "evidence_refs": refs, "valid_evidence_refs": valid_refs},
    }


def test_recheck_policy_is_gold_blind_and_capped():
    rows = [prediction(f"r{i}") for i in range(20)]
    rows[0]["trace_quality"]["valid_json"] = False
    rows[1]["trace_quality"]["valid_evidence_refs"] = 0
    selected = MODULE.select_rechecks(rows, "plain", 0.10)
    assert len(selected) == 2
    for row in rows:
        row["gold"] = "clean_safe" if row["gold"] != "clean_safe" else "attack_success"
    assert selected == MODULE.select_rechecks(rows, "plain", 0.10)


def test_recheck_instruction_does_not_copy_gold_or_full_target():
    row = {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": json.dumps({"run_evidence": {"observed": []}})},
        {"role": "assistant", "content": json.dumps({"decision": {"verdict": "attack_success"}})},
    ]}
    output = MODULE.add_recheck_instruction(row, prediction("r0"), ["invalid_json"])
    serialized = json.dumps(json.loads(output["messages"][1]["content"])["agent_recheck"])
    assert "gold" not in serialized
    assert "attack_success" not in serialized
    assert output["messages"][2] == row["messages"][2]


def test_cascade_disagreement_is_a_soft_recheck_signal():
    row = prediction("r", "attack_success")
    row.update({
        "structured_control": {"verdict": "attack_success", "scope": "global", "component_ids": ["G::run"]},
        "lm_verdict_pred_before_merge": "attack_failed",
        "lm_scope_pred_before_merge": "edge",
        "lm_components_pred_before_merge": ["E::a->b"],
    })
    score, reasons = MODULE.quality_signals(row, "cascade")
    assert score == 44
    assert set(reasons) == {"qwen_bert_verdict_disagreement", "qwen_bert_scope_disagreement", "qwen_bert_component_disagreement"}
