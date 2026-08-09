#!/usr/bin/env bash
set -euo pipefail

: "${BASE:?Set BASE to the experiment storage root}"
: "${REPO:?Set REPO to the Auditor-agent repository root}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-$BASE/qwen3_8b_sft_v19_final_once}"

for track in marble_only autogen_only mixed; do
  TRACK="$track" \
  ADAPTER="$BASE/sft_models/qwen3-8b-mas-auditor-lora-v19-$track" \
  OUTPUT="$RESULT_ROOT/$track" \
  PKG="$PKG" GPU="$GPU" \
    bash "$PKG/server_scripts/run_final_test_once_v19.sh"
done

python "$PKG/scripts/aggregate_three_track_metrics.py" \
  --marble "$RESULT_ROOT/marble_only/metrics.json" \
  --autogen "$RESULT_ROOT/autogen_only/metrics.json" \
  --mixed "$RESULT_ROOT/mixed/metrics.json" \
  --output "$RESULT_ROOT/three_track_average.json"
