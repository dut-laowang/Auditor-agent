#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
GPU="${GPU:-0}"

for track in marble_only autogen_only mixed; do
  adapter="$BASE/sft_models/qwen3-8b-mas-auditor-lora-v19-$track"
  validation="$BASE/qwen3_8b_sft_v19_${track}_validation"
  data="$PKG/three_track_datasets/$track"
  python "$PKG/scripts/audit_lexical_shortcuts.py" \
    --train-file "$data/train.jsonl" \
    --validation-file "$data/validation.jsonl" \
    --output "$data/lexical_shortcut_validation.json"
  if [[ ! -d "$data/validation_counterfactuals" ]]; then
    python "$PKG/scripts/make_validation_ablations.py" \
      --validation-file "$data/validation.jsonl" \
      --shortcut-report "$data/lexical_shortcut_validation.json" \
      --output-dir "$data/validation_counterfactuals" \
      --seed 42
  fi
  TRACK="$track" OUT="$adapter" BASE="$BASE" REPO="$REPO" GPU="$GPU" \
    bash "$PKG/server_scripts/run_train_v19.sh"
  TRACK="$track" ADAPTER="$adapter" OUTPUT="$validation" PKG="$PKG" GPU="$GPU" \
    bash "$PKG/server_scripts/run_validate_track_v19.sh"
done

echo "All three validation tracks completed. Freeze choices before final tests."
