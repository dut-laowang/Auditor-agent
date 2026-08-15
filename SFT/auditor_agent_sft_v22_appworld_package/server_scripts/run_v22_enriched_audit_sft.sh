#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
V22="$REPO/SFT/auditor_agent_sft_v22_appworld_package"
DATA="${V22_ENRICHED_DATA:-$BASE/v22_qwen32b_teacher_expansion_v4_full/expanded_audit_sft}"
RESULTS="${V22_ENRICHED_RESULTS:-$BASE/v22_appworld_enriched_audit_validation}"
INIT_ADAPTER="${V22_INIT_ADAPTER:-$MODEL_ROOT/qwen3-8b-v22-appworld-explainable-audit}"
ADAPTER="${V22_ENRICHED_ADAPTER:-$MODEL_ROOT/qwen3-8b-v22-appworld-audit-grade-v4}"
MODERN_EVAL="${MODERN_EVAL:-$BASE/v22_appworld_marble_validation/modernbert}"
EVAL="$RESULTS/qwen3_8b_enriched_audit"
TRAIN_SHA="df3a2cbf2e5a021a825e4f760f4a52877cfef8548e6e2ba7970266e0cd8f6e2c"
VALIDATION_SHA="e5fed0792445551b4162ebe2415c07e985efc43f7433e17896755e482a7c0596"

cd "$REPO"
mkdir -p "$RESULTS" "$MODEL_ROOT" "$EVAL"
test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$DATA/EXPANSION_CONTRACT.json"
test -f "$INIT_ADAPTER/run_manifest.json"
test -f "$MODERN_EVAL/predictions.jsonl"
test -f "$MODERN_EVAL/metrics.json"
[[ "$(sha256sum "$DATA/train.jsonl" | awk '{print $1}')" == "$TRAIN_SHA" ]]
[[ "$(sha256sum "$DATA/validation.jsonl" | awk '{print $1}')" == "$VALIDATION_SHA" ]]

# Continue the already validated V22 Audit LoRA on the same reports plus the
# three audit-grade fields. Checkpoints are resumable and kept in a new path.
if [[ ! -f "$ADAPTER/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --data-dir "$DATA" --output-dir "$ADAPTER" --init-adapter "$INIT_ADAPTER" \
    --max-len 8192 --epochs 2 --lr 1e-4 --batch "${QWEN_TRAIN_BATCH:-2}" \
    --grad-accum "${QWEN_GRAD_ACCUM:-8}" --seed 42 --resume auto \
    2>&1 | tee -a "$RESULTS/qwen_enriched_training.log"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$ADAPTER" --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$EVAL" --max-input-len 8192 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume \
  --structured-controls "$MODERN_EVAL/predictions.jsonl"

python "$V22/scripts/validate_v22_results.py" \
  --predictions "$EVAL/predictions.jsonl" --metrics "$EVAL/metrics.json" \
  --modernbert-metrics "$MODERN_EVAL/metrics.json" --output "$RESULTS/V22_CORE_QUALITY_GATE.json"
python "$V22/scripts/validate_v22_enriched_reports.py" \
  --predictions "$EVAL/predictions.jsonl" --validation "$DATA/validation.jsonl" \
  --output "$RESULTS/V22_ENRICHED_REPORT_QUALITY.json"

ARCHIVE="$BASE/$(basename "$RESULTS").tar.gz"
tar -czf "$ARCHIVE" -C "$BASE" "$(basename "$RESULTS")"
echo "DONE: $ARCHIVE"
