#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
GPU="${GPU:-0}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$RUN/base_dataset"
ADAPTER="$RUN/models/qwen3_8b_plain_sft"
CF="$OUT/counterfactual_data"
RESULTS="$OUT/counterfactual_results"

mkdir -p "$OUT" "$RESULTS" "$OUT/logs"
python "$V19/scripts/audit_lexical_shortcuts.py" --train-file "$DATA/train.jsonl" \
  --validation-file "$DATA/validation.jsonl" --output "$OUT/lexical_shortcut_validation.json"
if [[ ! -d "$CF" ]]; then
  python "$V19/scripts/make_validation_ablations.py" --validation-file "$DATA/validation.jsonl" \
    --shortcut-report "$OUT/lexical_shortcut_validation.json" --output-dir "$CF" --seed 42
fi
if [[ ! -f "$RESULTS/clean/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
    --mode sft --model Qwen/Qwen3-8B --adapter "$ADAPTER" --test-file "$DATA/validation.jsonl" \
    --dataset-role validation --output-dir "$RESULTS/clean" --max-input-len 12288 \
    --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp
fi
for file in "$CF"/*.jsonl; do
  name="$(basename "$file" .jsonl)"
  if [[ ! -f "$RESULTS/$name/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
      --mode sft --model Qwen/Qwen3-8B --adapter "$ADAPTER" --test-file "$file" \
      --dataset-role validation --output-dir "$RESULTS/$name" --max-input-len 12288 \
      --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp
  fi
done
python "$V19/scripts/summarize_counterfactuals.py" --result-root "$RESULTS" \
  --output "$RESULTS/counterfactual_summary.json"
