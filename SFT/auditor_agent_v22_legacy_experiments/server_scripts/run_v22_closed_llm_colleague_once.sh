#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(pwd)}"; REQUESTS="${V22_REQUESTS:?Set V22_REQUESTS to V22_CLOSED_LLM_REQUESTS.jsonl}"
PROVIDER="${PROVIDER:?Set PROVIDER=openai or anthropic}"; MODEL="${MODEL:?Set a pinned MODEL}"; OUT="${OUTPUT_DIR:-$PWD/v22_${PROVIDER}_${MODEL}_results}"
SCRIPT="$REPO/SFT/auditor_agent_v22_legacy_experiments/scripts/v22_closed_llm_baseline.py"
python "$SCRIPT" infer --provider "$PROVIDER" --model "$MODEL" --requests "$REQUESTS" --output-dir "$OUT" --workers "${API_WORKERS:-16}" --max-tokens "${API_MAX_TOKENS:-1024}"
tar -czf "$OUT/V22_${PROVIDER}_${MODEL}_RAW_RESULTS.tar.gz" -C "$OUT" INFERENCE_CONTRACT.json INFERENCE_COMPLETE.json api_predictions.jsonl
echo "RETURN THIS FILE: $OUT/V22_${PROVIDER}_${MODEL}_RAW_RESULTS.tar.gz"
