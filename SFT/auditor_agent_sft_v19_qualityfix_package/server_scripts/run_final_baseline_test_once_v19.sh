#!/usr/bin/env bash
set -euo pipefail

: "${BASE:=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
: "${BASELINE:?Set BASELINE=qwen32b or modernbert}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$PKG/three_track_datasets/marble_only"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
python "$PKG/scripts/restore_track_data.py" "$DATA"
(cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing)

case "$BASELINE" in
  qwen32b)
    MODEL="${MODEL:-Qwen/Qwen3-32B}"
    ADAPTER="${ADAPTER:-$BASE/sft_models/qwen3-32b-mas-auditor-qlora-v19-marble}"
    OUTPUT="${OUTPUT:-$BASE/qwen3_32b_v19_marble_final_test}"
    mkdir -p "$OUTPUT"
    CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema_v19.py" \
      --mode sft --model "$MODEL" --adapter "$ADAPTER" --load-in-4bit \
      --test-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
      --output-dir "$OUTPUT" --max-new-tokens 1024 --batch-size "${EVAL_BATCH_SIZE:-2}"
    ;;
  modernbert)
    MODEL="${MODEL:-answerdotai/ModernBERT-base}"
    MODEL_DIR="${MODEL_DIR:-$BASE/sft_models/modernbert-base-4096-multitask-v19-marble}"
    CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/checkpoint-epoch-3.pt}"
    OUTPUT="${OUTPUT:-$BASE/modernbert4096_v19_marble_final_test}"
    mkdir -p "$OUTPUT"
    CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/modernbert_multitask_v19.py" \
      --mode eval --model "$MODEL" --checkpoint "$CHECKPOINT" \
      --data-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
      --output-dir "$OUTPUT" --max-len 4096 --input-mode user \
      --batch "${EVAL_BATCH_SIZE:-2}" --seed 42
    ;;
  *) echo "BASELINE must be qwen32b or modernbert" >&2; exit 2 ;;
esac
