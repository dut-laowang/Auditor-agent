#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
echo "V22 legacy no-new-Qwen-SFT suite"
echo "source=$RUN output=$OUT GPU=${GPU:-0}"
python "$REPO/SFT/auditor_agent_v22_legacy_experiments/scripts/run_progress_suite.py" --repo "$REPO" --run-dir "$RUN" --out "$OUT"
tar -C "$OUT" -czf "$OUT/V22_LEGACY_FINAL_RESULTS.tar.gz" tables PROGRESS.json task_*.log heldout 2>/dev/null || \
tar -C "$OUT" -czf "$OUT/V22_LEGACY_FINAL_RESULTS.tar.gz" tables PROGRESS.json task_*.log
echo "TRANSFER THIS FILE: $OUT/V22_LEGACY_FINAL_RESULTS.tar.gz"
