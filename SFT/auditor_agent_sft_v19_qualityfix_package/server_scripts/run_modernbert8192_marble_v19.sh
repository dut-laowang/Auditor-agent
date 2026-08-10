#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$PKG/three_track_datasets/marble_only"
MODEL="answerdotai/ModernBERT-base"
OUT="${OUT:-$BASE/sft_models/modernbert-base-8192-multitask-v19-marble}"
RESULTS="${RESULTS:-$BASE/modernbert8192_v19_marble_validation}"
EPOCHS="${EPOCHS:-3}"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$OUT" "$RESULTS"

python "$PKG/scripts/check_baseline_environment.py" \
  --baseline modernbert | tee "$OUT/environment_preflight.json"
python "$PKG/scripts/selftest_baseline_logic.py" | tee "$OUT/logic_selftest.json"

python "$PKG/scripts/restore_track_data.py" "$DATA"
python "$PKG/scripts/audit_marble_baseline_contract.py" "$DATA" | tee "$OUT/data_contract_audit.json"

CHECKPOINT="$OUT/checkpoint-epoch-$EPOCHS.pt"
if [[ ! -f "$OUT/TRAINING_COMPLETE.json" ]]; then
  python "$PKG/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model "$MODEL" --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --data-file "$DATA/train.jsonl" \
    --dataset-role train --output-dir "$OUT" --max-len 8192 \
    --input-mode user --epochs "$EPOCHS" --lr "${LR:-2e-5}" \
    --batch "${TRAIN_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-8}" \
    --lambda-scope 1.0 --lambda-component 1.0 --seed 42 2>&1 | tee "$OUT/training.log"
fi

python "$PKG/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model "$MODEL" --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$CHECKPOINT" \
  --data-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$RESULTS" --max-len 8192 --input-mode user \
  --batch "${EVAL_BATCH_SIZE:-2}" --seed 42 2>&1 | tee "$RESULTS/evaluation.log"

echo "ModernBERT-8192 V19 MARBLE training and validation complete. Final test remains sealed."
