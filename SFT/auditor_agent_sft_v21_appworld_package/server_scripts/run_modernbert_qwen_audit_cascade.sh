#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

V20="${V20_RESULTS:-$BASE/v20_appworld_marble_core_validation}"
V21="${V21_RESULTS:-$BASE/v21_appworld_marble_validation}"
AUDIT_ADAPTER="${V21_AUDIT_ADAPTER:-$MODEL_ROOT/qwen3-8b-v21-appworld-conditional-audit}"
RESULTS="${CASCADE_RESULTS:-$BASE/v21_modernbert_qwen_appworld_validation}"
RUNTIME="$RESULTS/runtime_conditional"
EVAL_OUT="$RESULTS/modernbert_qwen_audit"
PKG="$REPO/SFT/auditor_agent_sft_v21_appworld_package"
EVAL="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/eval_qwen3_fullschema_v19.py"
CONTROLS="$V20/modernbert/predictions.jsonl"
REFERENCE="$V20/modernbert/metrics.json"
VALIDATION="$V20/context_filtered_dataset/validation.jsonl"
GOLD_TRAIN="$V21/dataset/conditional_gold/train.jsonl"

cd "$REPO"
mkdir -p "$RESULTS"
test -f "$AUDIT_ADAPTER/run_manifest.json"
test -f "$GOLD_TRAIN"
test -f "$CONTROLS"
test -f "$REFERENCE"

[[ "$(sha256sum "$VALIDATION" | awk '{print $1}')" == "5dd89c9950337ee277dedb203f0468ae754154c2c7af5d76eafc00514459805c" ]]
[[ "$(sha256sum "$CONTROLS" | awk '{print $1}')" == "eb5df953064098f181a597650d7e1a8f9ae6fb24f0ee45cba346bb7e0249aed5" ]]
[[ "$(sha256sum "$REFERENCE" | awk '{print $1}')" == "047f2cca4f4e081e666765f74e5dae8d75dca8261b90bb5948d15d3a90277287" ]]

# Gold controls exist only in the training split. Every validation control is an
# already-recorded ModernBERT prediction keyed by the frozen 406 validation IDs.
python "$PKG/scripts/materialize_predicted_validation.py" \
  --v20-validation "$VALIDATION" --controls "$CONTROLS" \
  --gold-train "$GOLD_TRAIN" --output-dir "$RUNTIME"

python "$EVAL" \
  --mode sft --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$AUDIT_ADAPTER" --test-file "$RUNTIME/validation.jsonl" \
  --dataset-role validation --output-dir "$EVAL_OUT" --max-input-len 8192 \
  --max-new-tokens 1024 --batch-size 4 --resume --structured-controls "$CONTROLS"

python - "$REFERENCE" "$EVAL_OUT/metrics.json" "$EVAL_OUT/predictions.jsonl" <<'PY'
import json, math, sys
reference = json.load(open(sys.argv[1]))
result = json.load(open(sys.argv[2]))
predictions = [json.loads(line) for line in open(sys.argv[3]) if line.strip()]
assert len(predictions) == result["n"] == reference["n"] == 406
paths = [
    ("three_class_accuracy",),
    ("three_class_report", "macro avg", "f1-score"),
    ("three_class_report", "attack_success", "recall"),
    ("localization", "component_micro_f1"),
    ("localization", "component_hit_rate"),
    ("localization", "component_exact_match"),
    ("localization", "scope_accuracy"),
]
def get(obj, path):
    for key in path:
        obj = obj[key]
    return obj
for path in paths:
    assert math.isclose(get(reference, path), get(result, path), rel_tol=0, abs_tol=1e-12), path
assert result["audit_trace_quality"]["valid_json_rate"] == 1.0
assert result["audit_trace_quality"]["has_audit_trace_rate"] == 1.0
assert result["audit_trace_quality"]["evidence_ref_validity_rate"] == 1.0
assert result["structured_controls"]["report_agreement_rate"] == 1.0
summary = {
    "dataset": "MARBLE x AppWorld", "dataset_role": "validation", "n": 406,
    "classification_and_localization_exactly_match_modernbert": True,
    "accuracy": result["three_class_accuracy"],
    "macro_f1": result["three_class_report"]["macro avg"]["f1-score"],
    "attack_success_recall": result["three_class_report"]["attack_success"]["recall"],
    "localization_f1": result["localization"]["component_micro_f1"],
    "valid_json_rate": result["audit_trace_quality"]["valid_json_rate"],
    "evidence_ref_validity_rate": result["audit_trace_quality"]["evidence_ref_validity_rate"],
    "sealed_test_accessed": False,
}
json.dump(summary, open(sys.argv[2].replace("metrics.json", "CASCADE_SUMMARY.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
PY

tar -czf "$BASE/v21_modernbert_qwen_appworld_validation.tar.gz" -C "$BASE" "$(basename "$RESULTS")"
echo "DONE: $BASE/v21_modernbert_qwen_appworld_validation.tar.gz"
