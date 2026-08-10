#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$PKG/three_track_datasets/marble_only"
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
OUT="${OUT:-$BASE/sft_models/qwen3-32b-mas-auditor-qlora-v19-marble}"
RESULTS="${RESULTS:-$BASE/qwen3_32b_v19_marble_validation}"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$OUT" "$RESULTS"

python "$PKG/scripts/restore_track_data.py" "$DATA"
(cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing)

if [[ ! -f "$OUT/TRAINING_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/train_qwen3_32b_qlora_v19.py" \
    --model "$MODEL" --data-dir "$DATA" --output-dir "$OUT" \
    --max-len 6144 --epochs 2 --lr 2e-4 \
    --batch "${TRAIN_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-8}" \
    --seed 42 --quantization 4bit --resume auto 2>&1 | tee "$OUT/training.log"
  touch "$OUT/TRAINING_COMPLETE"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model "$MODEL" --adapter "$OUT" --load-in-4bit \
  --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$RESULTS" --max-input-len 6144 --max-new-tokens 1024 \
  --batch-size "${EVAL_BATCH_SIZE:-2}" --resume 2>&1 | tee "$RESULTS/evaluation.log"

echo "Qwen3-32B V19 MARBLE training and validation complete. Final test remains sealed."
