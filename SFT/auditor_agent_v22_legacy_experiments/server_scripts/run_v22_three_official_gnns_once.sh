#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
SUP="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
PKG="$REPO/SFT/auditor_agent_v22_legacy_experiments"
start="$(date +%s)"
echo "[1/3] G-Safeguard official encoder + V22 heads"
GNN_METHODS=gat bash "$PKG/server_scripts/run_v22_official_gsafe_tam_once.sh"
echo "[2/3] TAM official encoder + V22 heads"
GNN_METHODS=tam bash "$PKG/server_scripts/run_v22_official_gsafe_tam_once.sh"
echo "[3/3] XG-Guard official OursMethod + V22 adapter"
bash "$PKG/server_scripts/run_v22_official_xgguard_once.sh"
python "$PKG/scripts/render_experiment_tables.py" --run-dir "$RUN" --supplement-dir "$SUP" --output-dir "$SUP/tables"
core="$SUP/V22_THREE_OFFICIAL_GNN_RESULTS.tar.gz"
(
 cd "$SUP"
 find baselines/gsafeguard_official_v22_v1 baselines/tam_official_v22_v1 baselines/xgguard_official_v22_v1 tables \
   -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.tex' \) ! -path '*/cache/*' -print0 |
   tar --null --files-from=- -czf "$core"
)
echo "ALL THREE COMPLETE in $(( $(date +%s)-start )) seconds"
echo "TRANSFER THIS FILE: $core"
