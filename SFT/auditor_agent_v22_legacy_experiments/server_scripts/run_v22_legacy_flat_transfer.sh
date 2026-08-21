#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
GPU="${GPU:-0}"
V18="$REPO/SFT/auditor_agent_sft_v18_single_agent_baselines_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
FLAT="$OUT/single_transfer/v22_flat_data"
MODEL="$OUT/single_transfer/qwen3_8b_flat_model"
EVAL="$OUT/single_transfer/qwen3_8b_flat_validation"
REV="b968826d9c46dd6066d109eabc6255188de91218"

[[ "$(wc -l < "$RUN/base_dataset/train.jsonl" | tr -d ' ')" -eq 10438 ]]
[[ "$(wc -l < "$RUN/base_dataset/validation.jsonl" | tr -d ' ')" -eq 2954 ]]
mkdir -p "$OUT/single_transfer" "$OUT/logs"
python "$V18/scripts/build_v18_flat_dataset.py" --graph-data-dir "$RUN/base_dataset" \
  --output-dir "$FLAT" --eval-split validation

if [[ ! -f "$MODEL/adapter_model.safetensors" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision "$REV" --data-dir "$FLAT" --output-dir "$MODEL" \
    --max-len 12288 --context-contract v22-flat-12288 --prompt-overflow error \
    --epochs "${QWEN_EPOCHS:-2}" --lr "${QWEN_LR:-2e-4}" \
    --batch "${QWEN_TRAIN_BATCH:-1}" --grad-accum "${QWEN_GRAD_ACCUM:-16}" \
    --seed 42 --resume auto --disable-cudnn-sdp 2>&1 | tee -a "$OUT/logs/v22_flat_train.log"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision "$REV" --adapter "$MODEL" \
  --test-file "$FLAT/validation.jsonl" --dataset-role validation --output-dir "$EVAL" \
  --max-input-len 12288 --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" \
  --resume --disable-cudnn-sdp 2>&1 | tee -a "$OUT/logs/v22_flat_validation.log"
