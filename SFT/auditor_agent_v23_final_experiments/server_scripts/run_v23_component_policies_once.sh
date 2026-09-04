#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"; REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"; OUT="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}/components"
Q="$RUN/qwen3_8b_plain_sft_test/predictions.jsonl"; B="$RUN/modernbert_test/predictions.jsonl"; IDX="$RUN/modernbert_data/test_track_index.jsonl"
for f in "$Q" "$B" "$IDX"; do [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }; done
export PYTHONPATH="$REPO/SFT/auditor_agent_sft_v22_all_package/scripts${PYTHONPATH:+:$PYTHONPATH}"
python "$REPO/SFT/auditor_agent_v23_final_experiments/scripts/evaluate_component_policies.py" --qwen "$Q" --bert "$B" --index "$IDX" --output-dir "$OUT" --max-verify-rate "${RULE_VERIFY_RATE:-0.15}"
echo "DONE: $OUT/COMPONENT_POLICIES_COMPLETE.json"
