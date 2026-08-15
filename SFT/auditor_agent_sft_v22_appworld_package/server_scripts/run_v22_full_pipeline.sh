#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
V20="$REPO/SFT/auditor_agent_sft_v20_appworld_marble_package"
V22="$REPO/SFT/auditor_agent_sft_v22_appworld_package"
V20_RESULTS="${V20_RESULTS:-$BASE/v20_appworld_marble_core_validation}"
SOURCE_DATA="$V20_RESULTS/context_filtered_dataset"
RESULTS="${V22_RESULTS:-$BASE/v22_appworld_marble_validation}"
DATA="$RESULTS/dataset"
MODERN_EVAL="$RESULTS/modernbert"
MODERN_MODEL="${MODERN_MODEL:-$MODEL_ROOT/modernbert-base-8192-sdpa-fp32-multitask-v20-appworld-marble-bertfiltered}"
V20_QWEN="${V20_QWEN_ADAPTER:-$MODEL_ROOT/qwen3-8b-mas-auditor-lora-v20-appworld-marble}"
V22_QWEN="${V22_QWEN_ADAPTER:-$MODEL_ROOT/qwen3-8b-v22-appworld-explainable-audit}"
QWEN_EVAL="$RESULTS/qwen3_8b_explainable_audit"
TRAIN_SHA="20372c1d2dad08be7d43465d0d4887491ec82b2d9eca8fff61d16a4708124145"
VALIDATION_SHA="5dd89c9950337ee277dedb203f0468ae754154c2c7af5d76eafc00514459805c"

cd "$REPO"
mkdir -p "$RESULTS" "$MODEL_ROOT" "$MODERN_EVAL"
test -f "$SOURCE_DATA/train.jsonl"
test -f "$SOURCE_DATA/validation.jsonl"
test -f "$V20_QWEN/run_manifest.json"
[[ "$(sha256sum "$SOURCE_DATA/train.jsonl" | awk '{print $1}')" == "$TRAIN_SHA" ]]
[[ "$(sha256sum "$SOURCE_DATA/validation.jsonl" | awk '{print $1}')" == "$VALIDATION_SHA" ]]

# Inspector: train only if the existing controlled V20 model is unavailable;
# always rerun prediction on the exact frozen 406 V22 validation IDs.
if [[ ! -f "$MODERN_MODEL/TRAINING_COMPLETE.json" ]]; then
  mkdir -p "$MODERN_MODEL"
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --data-file "$SOURCE_DATA/train.jsonl" --dataset-role train --output-dir "$MODERN_MODEL" \
    --expected-train-sha256 "$TRAIN_SHA" --expected-validation-sha256 "$VALIDATION_SHA" \
    --max-len 8192 --attn-implementation sdpa --input-mode user --epochs 3 --lr 2e-5 \
    --batch "${MODERN_TRAIN_BATCH:-2}" --grad-accum "${MODERN_GRAD_ACCUM:-8}" \
    --lambda-scope 1.0 --lambda-component 1.0 --seed 42 \
    2>&1 | tee "$RESULTS/modernbert_training.log"
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$MODERN_MODEL/checkpoint-epoch-3.pt" \
  --data-file "$SOURCE_DATA/validation.jsonl" --dataset-role validation \
  --expected-train-sha256 "$TRAIN_SHA" --expected-validation-sha256 "$VALIDATION_SHA" \
  --output-dir "$MODERN_EVAL" --max-len 8192 --attn-implementation sdpa \
  --input-mode user --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 \
  2>&1 | tee "$MODERN_EVAL/evaluation.log"

# Deterministic Trajectory Adapter. It also hard-gates the exact shared IDs and
# injects train-gold controls versus validation-predicted controls.
python "$V22/scripts/build_v22_audit_dataset.py" \
  --source-data "$SOURCE_DATA" --modernbert-predictions "$MODERN_EVAL/predictions.jsonl" \
  --output-dir "$DATA"

# Conditional explainable Audit SFT. Initialize from V20 audit capabilities;
# checkpoints are saved every 100 optimizer steps and resume automatically.
if [[ ! -f "$V22_QWEN/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --data-dir "$DATA/audit_sft" --output-dir "$V22_QWEN" --init-adapter "$V20_QWEN" \
    --max-len 8192 --epochs 2 --lr 2e-4 --batch 2 --grad-accum 8 --seed 42 --resume auto \
    2>&1 | tee "$RESULTS/qwen_v22_training.log"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$V22_QWEN" --test-file "$DATA/audit_sft/validation.jsonl" \
  --dataset-role validation --output-dir "$QWEN_EVAL" --max-input-len 8192 \
  --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" --resume \
  --structured-controls "$MODERN_EVAL/predictions.jsonl"

python "$V22/scripts/validate_v22_results.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --metrics "$QWEN_EVAL/metrics.json" \
  --modernbert-metrics "$MODERN_EVAL/metrics.json" --output "$RESULTS/V22_QUALITY_GATE.json"

tar -czf "$BASE/v22_appworld_marble_validation.tar.gz" -C "$BASE" "$(basename "$RESULTS")"
echo "DONE: $BASE/v22_appworld_marble_validation.tar.gz"
