#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_v22_legacy_experiments"

# Default: cheap, publication-useful held-out pilot on ModernBERT.
bash "$PKG/server_scripts/run_v22_legacy_supplement_suite.sh"

# Optional public graph baseline. It uses the same frozen legacy train/validation.
if [[ "${RUN_EXTERNAL_BASELINES:-0}" == 1 ]]; then
  bash "$PKG/server_scripts/run_v22_legacy_external_baselines.sh"
fi

# Optional controlled V22 same-ID Flat-vs-Graph transfer experiment.
if [[ "${RUN_SINGLE_TRANSFER:-0}" == 1 ]]; then
  bash "$PKG/server_scripts/run_v22_legacy_flat_transfer.sh"
fi

if [[ "${RUN_COUNTERFACTUALS:-0}" == 1 ]]; then
  bash "$PKG/server_scripts/run_v22_legacy_counterfactuals.sh"
fi

python "$PKG/scripts/render_experiment_tables.py" \
  --run-dir "${V22_LEGACY_RUN:-$BASE/v22_all_run}" \
  --supplement-dir "${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}" \
  --output-dir "${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}/tables"
