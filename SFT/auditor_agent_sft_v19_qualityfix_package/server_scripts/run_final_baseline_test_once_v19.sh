#!/usr/bin/env bash
set -euo pipefail

: "${BASE:=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
: "${BASELINE:?Set BASELINE=qwen32b or modernbert}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$PKG/three_track_datasets/marble_only"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
python "$PKG/scripts/restore_track_data.py" "$DATA"
(cd "$DATA" && sha256sum -c SHA256SUMS --ignore-missing)
python "$PKG/scripts/audit_marble_baseline_contract.py" "$DATA"

case "$BASELINE" in
  qwen32b)
    python "$PKG/scripts/check_baseline_environment.py" --baseline qwen32b
    python "$PKG/scripts/selftest_baseline_logic.py"
    MODEL="Qwen/Qwen3-32B"
    ADAPTER="${ADAPTER:-$BASE/sft_models/qwen3-32b-mas-auditor-qlora-v19-marble}"
    OUTPUT="${OUTPUT:-$BASE/qwen3_32b_v19_marble_final_test}"
    mkdir -p "$OUTPUT"
    python "$PKG/server_scripts/eval_qwen3_fullschema_v19.py" \
      --mode sft --model "$MODEL" --revision 9216db5781bf21249d130ec9da846c4624c16137 \
      --adapter "$ADAPTER" --load-in-4bit \
      --test-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
      --output-dir "$OUTPUT" --max-input-len 6144 --max-new-tokens 1024 \
      --batch-size "${EVAL_BATCH_SIZE:-2}"
    ;;
  modernbert)
    python "$PKG/scripts/check_baseline_environment.py" --baseline modernbert
    python "$PKG/scripts/selftest_baseline_logic.py"
    MODEL="answerdotai/ModernBERT-base"
    MODEL_DIR="${MODEL_DIR:-$BASE/sft_models/modernbert-base-6144-multitask-v19-marble}"
    CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/checkpoint-epoch-3.pt}"
    OUTPUT="${OUTPUT:-$BASE/modernbert6144_v19_marble_final_test}"
    mkdir -p "$OUTPUT"
    python "$PKG/server_scripts/modernbert_multitask_v19.py" \
      --mode eval --model "$MODEL" --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
      --checkpoint "$CHECKPOINT" \
      --data-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
      --output-dir "$OUTPUT" --max-len 6144 --input-mode user \
      --batch "${EVAL_BATCH_SIZE:-2}" --seed 42
    ;;
  *) echo "BASELINE must be qwen32b or modernbert" >&2; exit 2 ;;
esac
