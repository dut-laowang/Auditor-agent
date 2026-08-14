#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
V20="$REPO/SFT/auditor_agent_sft_v20_appworld_marble_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
SOURCE_RESULTS="${SOURCE_RESULTS:-$BASE/v20_appworld_marble_core_validation}"
DATA="$SOURCE_RESULTS/context_filtered_dataset"
RESULTS="${RESULTS:-$BASE/v20_appworld_marble_joint_sft_validation}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
ADAPTER="$MODEL_ROOT/qwen3-8b-v20-appworld-marble-joint-audit-v1"
GPU="${GPU:-0}"

export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RESULTS" "$MODEL_ROOT"

test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$SOURCE_RESULTS/qwen3_8b_clean/metrics.json"
EXPECTED_TRAIN_SHA="20372c1d2dad08be7d43465d0d4887491ec82b2d9eca8fff61d16a4708124145"
EXPECTED_VALIDATION_SHA="5dd89c9950337ee277dedb203f0468ae754154c2c7af5d76eafc00514459805c"
EXPECTED_BASELINE_SHA="be47dd725bfb09ae197f778c2eb46bd666465a8ead87efe0bf730ba819dfeb2f"
[[ "$(sha256sum "$DATA/train.jsonl" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA" ]]
[[ "$(sha256sum "$DATA/validation.jsonl" | awk '{print $1}')" == "$EXPECTED_VALIDATION_SHA" ]]
[[ "$(sha256sum "$SOURCE_RESULTS/qwen3_8b_clean/metrics.json" | awk '{print $1}')" == "$EXPECTED_BASELINE_SHA" ]]

if [[ ! -f "$ADAPTER/TRAINING_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V20/server_scripts/train_qwen3_joint_audit_sft.py" \
    --data-dir "$DATA" --output-dir "$ADAPTER" --max-len 8192 \
    --epochs 2 --lr 2e-4 --batch 2 --grad-accum 8 --seed 42 \
    --lambda-cls "${LAMBDA_CLS:-0.5}" --lambda-loc "${LAMBDA_LOC:-2.0}" --resume auto \
    2>&1 | tee "$RESULTS/training.log"
  touch "$ADAPTER/TRAINING_COMPLETE"
fi

# Deliberately reuse the frozen V20 evaluator.  The auxiliary head never sees
# validation gold labels and never changes decoding or the external JSON schema.
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --adapter "$ADAPTER" \
  --verdict-head "$ADAPTER/verdict_head.pt" \
  --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$RESULTS/qwen3_8b_joint_clean" --max-new-tokens 1024 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume

python "$V20/server_scripts/compare_joint_sft.py" \
  --baseline "$SOURCE_RESULTS/qwen3_8b_clean/metrics.json" \
  --joint "$RESULTS/qwen3_8b_joint_clean/metrics.json" \
  --output "$RESULTS/baseline_vs_joint.json"
python "$V19/scripts/write_sha256_manifest.py" "$RESULTS"
touch "$RESULTS/VALIDATION_COMPLETE"
tar -czf "$RESULTS.tar.gz" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")"
echo "Joint SFT validation complete. Sealed test was not accessed."
echo "Comparison: $RESULTS/baseline_vs_joint.json"
