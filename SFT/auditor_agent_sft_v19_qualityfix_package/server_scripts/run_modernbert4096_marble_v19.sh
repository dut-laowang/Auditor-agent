#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$PKG/three_track_datasets/marble_only"
GPU="${GPU:-0}"
MODEL="${MODEL:-answerdotai/ModernBERT-base}"
OUT="${OUT:-$BASE/sft_models/modernbert-base-4096-multitask-v19-marble}"
RESULTS="${RESULTS:-$BASE/modernbert4096_v19_marble_validation}"
EPOCHS="${EPOCHS:-3}"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$OUT" "$RESULTS"

python "$PKG/scripts/restore_track_data.py" "$DATA"
(cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing)

CHECKPOINT="$OUT/checkpoint-epoch-$EPOCHS.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model "$MODEL" --data-file "$DATA/train.jsonl" \
    --dataset-role train --output-dir "$OUT" --max-len 4096 \
    --input-mode user --epochs "$EPOCHS" --lr "${LR:-2e-5}" \
    --batch "${TRAIN_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-8}" \
    --lambda-scope 1.0 --lambda-component 1.0 --seed 42 2>&1 | tee "$OUT/training.log"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --data-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$RESULTS" --max-len 4096 --input-mode user \
  --batch "${EVAL_BATCH_SIZE:-2}" --seed 42 2>&1 | tee "$RESULTS/evaluation.log"

echo "ModernBERT-4096 V19 MARBLE training and validation complete. Final test remains sealed."
