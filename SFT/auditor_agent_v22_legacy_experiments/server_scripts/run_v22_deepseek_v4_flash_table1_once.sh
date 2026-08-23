#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
OUT="${DEEPSEEK_V4_OUT:-$RUN/deepseek_v4_flash_table1_test}"
DATA="$RUN/modernbert_sealed_test_source/test.jsonl"
SCRIPT="$REPO/SFT/auditor_agent_v22_legacy_experiments/scripts/v22_deepseek_v4_flash_baseline.py"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${EVAL_API_KEY:-}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-${EVAL_BASE_URL:-https://api.deepseek.com}}"
[[ -n "$DEEPSEEK_API_KEY" ]] || { echo "Set DEEPSEEK_API_KEY or reuse SGM's EVAL_API_KEY" >&2; exit 2; }
[[ -f "$DATA" && -f "$SCRIPT" ]] || { echo "Missing V22 test data or evaluator" >&2; exit 2; }
mkdir -p "$OUT"

echo "DeepSeek V4-Flash Table-1 baseline"
echo "rows=2539 workers=${DEEPSEEK_WORKERS:-32} max_output=${DEEPSEEK_MAX_TOKENS:-1024}"
echo "Progress shows completed rows, failures, live estimated USD, output tokens, speed and ETA."

python "$SCRIPT" infer --data "$DATA" --output-dir "$OUT" \
  --model deepseek-v4-flash --base-url "$DEEPSEEK_BASE_URL" \
  --workers "${DEEPSEEK_WORKERS:-32}" --max-tokens "${DEEPSEEK_MAX_TOKENS:-1024}" \
  --max-retries "${DEEPSEEK_MAX_RETRIES:-5}"
python "$SCRIPT" score --data "$DATA" --output-dir "$OUT"

tar -czf "$OUT/V22_DEEPSEEK_V4_FLASH_TABLE1_RESULTS.tar.gz" -C "$OUT" \
  INFERENCE_CONTRACT.json INFERENCE_COMPLETE.json metrics.json scored_predictions.jsonl RUN_COMPLETE.json
echo "TRANSFER: $OUT/V22_DEEPSEEK_V4_FLASH_TABLE1_RESULTS.tar.gz"
