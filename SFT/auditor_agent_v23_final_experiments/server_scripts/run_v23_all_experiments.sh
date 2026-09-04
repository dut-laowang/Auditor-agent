#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}";REPO="${REPO:-$BASE/Auditor-agent}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR to v23_final_aligned_combined}";RUN="${V23_RUN:-$BASE/v23_final_run}";OUT="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}"
python "$REPO/SFT/auditor_agent_v23_final_experiments/scripts/run_v23_experiment_suite.py" --repo "$REPO" --data "$DATA" --run "$RUN" --experiments "$OUT"
ARCHIVE="${V23_RESULTS_ARCHIVE:-${OUT%/}_V23_FINAL_EXPERIMENT_RESULTS.tar.gz}"
tar -czf "$ARCHIVE" -C "$OUT" --exclude='*.pt' --exclude='*.safetensors' .
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "TRANSFER: $ARCHIVE"
