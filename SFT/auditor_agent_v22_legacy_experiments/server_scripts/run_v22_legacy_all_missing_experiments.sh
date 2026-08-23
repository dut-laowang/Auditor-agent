#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run_v3}"
echo "V22 legacy no-new-Qwen-SFT suite"
echo "source=$RUN output=$OUT GPU=${GPU:-0}"
echo "agent_common_eligible_rows=2531"
python "$REPO/SFT/auditor_agent_v22_legacy_experiments/scripts/run_progress_suite.py" --repo "$REPO" --run-dir "$RUN" --out "$OUT"
mkdir -p "$OUT/collected_results/agent"
for f in AGENT_TEST_COMPARISON.json AGENT_TEST_COMPLETE.json PREPARE_MANIFEST.json CALIBRATION_POLICY.json; do
  src="$RUN/plain_hetero_agent_full_test_common2531_v3/$f"
  [[ ! -f "$src" ]] || cp -f "$src" "$OUT/collected_results/agent/$f"
done
(
  cd "$OUT"
  filelist="CORE_RESULT_FILELIST.txt"
  : > "$filelist"
  for f in PROGRESS.json SUPPLEMENT_SUITE_COMPLETE.json; do
    [[ ! -f "$f" ]] || printf '%s\0' "$f" >> "$filelist"
  done
  for d in tables heldout baselines collected_results; do
    [[ ! -d "$d" ]] || find "$d" -type f \
      \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.tex' \) \
      ! -path '*/cache/*' -print0 >> "$filelist"
  done
  tar --null --files-from="$filelist" -czf V22_LEGACY_CORE_RESULTS.tar.gz
)
echo "TRANSFER THIS FILE: $OUT/V22_LEGACY_CORE_RESULTS.tar.gz"
echo "Model checkpoints remain on server and are intentionally excluded."
