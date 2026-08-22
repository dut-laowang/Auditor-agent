#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
REPO="${REPO:-$BASE/Auditor-agent}"

# Full test means the frozen Qwen/ModernBERT common eligible test subset.
# It is 2,531 rows (2,539 sealed rows minus 8 ModernBERT context-ineligible
# rows), not validation+test combined.
AGENT_TEST_ROWS=2531 \
AGENT_OUTPUT_DIR="$RUN/plain_hetero_agent_full_test_common2531_v3" \
bash "$REPO/SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_plain_hetero_agent_test300_once.sh"
