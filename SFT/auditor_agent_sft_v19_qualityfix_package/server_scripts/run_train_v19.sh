#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
TRACK="${TRACK:-mixed}"
case "$TRACK" in marble_only|autogen_only|mixed) ;; *) echo "Invalid TRACK=$TRACK" >&2; exit 2;; esac
DATA="$PKG/three_track_datasets/$TRACK"
OUT="${OUT:-$BASE/sft_models/qwen3-8b-mas-auditor-lora-v19-$TRACK}"
GPU="${GPU:-0}"

test -f "$DATA/train.jsonl"
test -f "$DATA/validation.jsonl"
test -f "$DATA/test.jsonl"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

if [[ -f "$OUT/TRAINING_COMPLETE" ]]; then
  echo "Training already complete: $OUT"
else
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --max-len 8192 \
    --epochs 2 \
    --lr 2e-4 \
    --batch 2 \
    --grad-accum 8 \
    --seed 42 \
    --resume auto 2>&1 | tee "$OUT/training.log"
  touch "$OUT/TRAINING_COMPLETE"
fi

echo "Training complete. Do not run final test until validation and ablations are frozen."
