#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(pwd)}"
ASSET="$REPO/SFT/auditor_agent_v22_legacy_experiments/assets/V22_CLOSED_LLM_INPUTS_LABEL_BLIND.tar.gz"
INPUT_CACHE="${V22_INPUT_CACHE:-$REPO/.v22_closed_llm_inputs}"
EXPECTED_REQUESTS_SHA256="5db62d9bcf2fae1d8a1a14ac0f492af0a68a4eb44293554bd961d2f6818d3dae"

if [[ -n "${V22_REQUESTS:-}" ]]; then
  REQUESTS="$V22_REQUESTS"
else
  [[ -f "$ASSET" ]] || { echo "Missing bundled label-blind input asset: $ASSET" >&2; exit 1; }
  mkdir -p "$INPUT_CACHE"
  REQUESTS="$INPUT_CACHE/V22_CLOSED_LLM_REQUESTS.jsonl"
  if [[ ! -f "$REQUESTS" ]]; then
    tar -xzf "$ASSET" -C "$INPUT_CACHE"
  fi
fi

[[ -f "$REQUESTS" ]] || { echo "Missing requests file: $REQUESTS" >&2; exit 1; }
ACTUAL_REQUESTS_SHA256="$(sha256sum "$REQUESTS" | awk '{print $1}')"
[[ "$ACTUAL_REQUESTS_SHA256" == "$EXPECTED_REQUESTS_SHA256" ]] || {
  echo "Request SHA-256 mismatch: expected $EXPECTED_REQUESTS_SHA256, got $ACTUAL_REQUESTS_SHA256" >&2
  exit 1
}
REQUEST_ROWS="$(wc -l < "$REQUESTS" | tr -d ' ')"
[[ "$REQUEST_ROWS" == "2539" ]] || { echo "Expected 2539 requests, got $REQUEST_ROWS" >&2; exit 1; }

PROVIDER="${PROVIDER:?Set PROVIDER=openai or anthropic}"; MODEL="${MODEL:?Set a pinned MODEL}"; OUT="${OUTPUT_DIR:-$PWD/v22_${PROVIDER}_${MODEL}_results}"
SCRIPT="$REPO/SFT/auditor_agent_v22_legacy_experiments/scripts/v22_closed_llm_baseline.py"
echo "V22 input contract PASS: rows=$REQUEST_ROWS sha256=$ACTUAL_REQUESTS_SHA256"
python "$SCRIPT" infer --provider "$PROVIDER" --model "$MODEL" --requests "$REQUESTS" --output-dir "$OUT" --workers "${API_WORKERS:-16}" --max-tokens "${API_MAX_TOKENS:-1024}"
tar -czf "$OUT/V22_${PROVIDER}_${MODEL}_RAW_RESULTS.tar.gz" -C "$OUT" INFERENCE_CONTRACT.json INFERENCE_COMPLETE.json api_predictions.jsonl
echo "RETURN THIS FILE: $OUT/V22_${PROVIDER}_${MODEL}_RAW_RESULTS.tar.gz"
