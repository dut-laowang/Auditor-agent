#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
# Reuse the exact shared cache used by the V20 AppWorld pipeline. Without these
# exports, Hugging Face falls back to the small login-node home cache and tries
# to download the already available Qwen weights again.
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
V20_RESULTS="${V20_RESULTS:-$BASE/v20_appworld_marble_core_validation}"
V20_DATA="$V20_RESULTS/context_filtered_dataset"
V20_ADAPTER="${V20_ADAPTER:-$MODEL_ROOT/qwen3-8b-mas-auditor-lora-v20-appworld-marble}"
RESULTS="${V21_RESULTS:-$BASE/v21_appworld_marble_validation}"
DATASET="$RESULTS/dataset"
RUNTIME="$RESULTS/runtime_conditional"
HEADS="${V21_HEADS:-$MODEL_ROOT/qwen3-8b-v21-appworld-frozen-heads}"
HEAD_CACHE="${V21_HEAD_CACHE:-$MODEL_ROOT/qwen3-8b-v21-appworld-head-cache}"
AUDIT_ADAPTER="${V21_AUDIT_ADAPTER:-$MODEL_ROOT/qwen3-8b-v21-appworld-conditional-audit}"
EVAL_OUT="$RESULTS/qwen3_8b_v21"
PKG="$REPO/SFT/auditor_agent_sft_v21_appworld_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts"

cd "$REPO"
mkdir -p "$RESULTS" "$MODEL_ROOT"
test -f "$V20_DATA/train.jsonl"
test -f "$V20_DATA/validation.jsonl"
test -f "$V20_ADAPTER/run_manifest.json"

python "$PKG/scripts/build_v21_views.py" \
  --v20-data "$V20_DATA" --output-dir "$DATASET"

if [[ ! -f "$HEADS/TRAINING_COMPLETE.json" ]]; then
  python "$PKG/server_scripts/qwen_frozen_multitask_heads_v21.py" \
    --train-file "$DATASET/discriminative/train.jsonl" \
    --validation-file "$DATASET/discriminative/validation.jsonl" \
    --audit-adapter "$V20_ADAPTER" --output-dir "$HEADS" --cache-dir "$HEAD_CACHE" \
    --epochs 15 --batch 32 --candidate-batch 32 --lr 3e-4 --seed 42
fi

python "$PKG/scripts/materialize_predicted_validation.py" \
  --v20-validation "$V20_DATA/validation.jsonl" \
  --controls "$HEADS/validation_controls.jsonl" \
  --gold-train "$DATASET/conditional_gold/train.jsonl" --output-dir "$RUNTIME"

if [[ ! -f "$AUDIT_ADAPTER/run_manifest.json" ]]; then
  python "$V19/train_qwen3_lora_sft_v19.py" \
    --data-dir "$RUNTIME" --output-dir "$AUDIT_ADAPTER" \
    --init-adapter "$V20_ADAPTER" --model Qwen/Qwen3-8B \
    --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --max-len 8192 --epochs 2 --lr 2e-4 --batch 2 --grad-accum 8 \
    --seed 42 --resume auto
fi

python "$V19/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$AUDIT_ADAPTER" --test-file "$RUNTIME/validation.jsonl" \
  --dataset-role validation --output-dir "$EVAL_OUT" --max-input-len 8192 \
  --max-new-tokens 1024 --batch-size 4 --resume \
  --structured-controls "$HEADS/validation_controls.jsonl"

python "$PKG/scripts/summarize_v21.py" \
  --v20-qwen "$V20_RESULTS/qwen3_8b_clean/metrics.json" \
  --v20-modernbert "$V20_RESULTS/modernbert/metrics.json" \
  --v21 "$EVAL_OUT/metrics.json" --head-metrics "$HEADS/metrics.json" \
  --output "$RESULTS/V21_COMPARISON.json"

python - "$EVAL_OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["n"] == 406
assert m["dataset_role"] == "validation"
assert m["audit_trace_quality"]["valid_json_rate"] == 1.0
assert m["audit_trace_quality"]["evidence_ref_validity_rate"] == 1.0
assert m["structured_controls"]["report_agreement_rate"] == 1.0
print("V21 quality gate: PASS")
PY

tar -czf "$BASE/v21_appworld_marble_validation.tar.gz" -C "$BASE" "$(basename "$RESULTS")"
echo "DONE: $BASE/v21_appworld_marble_validation.tar.gz"
