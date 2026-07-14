#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}
REPO=${REPO:-$BASE/Auditor-agent}
PKG=${PKG:-$REPO/SFT/auditor_agent_gnn_baselines_package}
V12_PKG=${V12_PKG:-$REPO/SFT/auditor_agent_sft_v12_package}
V12_DATA=${V12_DATA:-$V12_PKG/sft_dataset_graph_grounded_v12}
BALANCED50=${BALANCED50:-$BASE/balanced_common_v8_v9_50/balanced_common_run_ids.txt}
MODEL=${MODEL:-gpt-4o-mini}
OUT=${OUT:-$BASE/openai_vs_sft_common50}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-1024}

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 1
fi

cd "$REPO"
git pull
cd "$BASE"

python - <<'PY'
import importlib.util
import subprocess
import sys

missing = [pkg for pkg in ["openai", "sklearn", "tqdm"] if importlib.util.find_spec(pkg) is None]
if "sklearn" in missing:
    missing.remove("sklearn")
    missing.append("scikit-learn")
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

mkdir -p "$OUT"
SUBSET="$OUT/v12_common50.jsonl"
OPENAI_OUT="$OUT/$MODEL"

python "$V12_PKG/server_scripts/make_subset_by_ids.py" \
  --all-file "$V12_DATA/all.jsonl" \
  --ids-file "$BALANCED50" \
  --output-file "$SUBSET"

rm -rf "$OPENAI_OUT"
python "$PKG/server_scripts/eval_openai_fullschema.py" \
  --model "$MODEL" \
  --test-file "$SUBSET" \
  --output-dir "$OPENAI_OUT" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS"

for POLICY in strict_agent agent_or_tool_owner; do
  POLICY_OUT="$OUT/$POLICY"
  mkdir -p "$POLICY_OUT"

  GNN_BASE="$BASE/gnn_vs_sft_common50/$POLICY"
  if [ ! -f "$GNN_BASE/gsafeguard/metrics.json" ]; then
    GNN_BASE="$PKG/results/common50_latest/$POLICY"
  fi

  python "$PKG/server_scripts/compare_gnn_vs_sft.py" \
    --label-policy "$POLICY" \
    --v8-metrics "$BASE/qwen3_8b_sft_v8_balanced50_eval_tok1024/metrics.json" \
    --v8-predictions "$BASE/qwen3_8b_sft_v8_balanced50_eval_tok1024/predictions.jsonl" \
    --v12-metrics "$BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/metrics.json" \
    --v12-predictions "$BASE/qwen3_8b_sft_v12_balanced50_eval_tok1024/predictions.jsonl" \
    --openai-name "$MODEL" \
    --openai-metrics "$OPENAI_OUT/metrics.json" \
    --openai-predictions "$OPENAI_OUT/predictions.jsonl" \
    --gsafeguard-metrics "$GNN_BASE/gsafeguard/metrics.json" \
    --gsafeguard-predictions "$GNN_BASE/gsafeguard/predictions.jsonl" \
    --blindguard-metrics "$GNN_BASE/blindguard/metrics.json" \
    --blindguard-predictions "$GNN_BASE/blindguard/predictions.jsonl" \
    --output "$POLICY_OUT/comparison_table.json"
done

echo "Done."
echo "OpenAI predictions: $OPENAI_OUT/predictions.jsonl"
echo "OpenAI metrics: $OPENAI_OUT/metrics.json"
echo "Strict-agent comparison: $OUT/strict_agent/comparison_table.json"
echo "Agent-or-tool-owner comparison: $OUT/agent_or_tool_owner/comparison_table.json"
