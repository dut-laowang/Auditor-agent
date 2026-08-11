#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_gnn_baselines_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="$V19/three_track_datasets/marble_only"
REFS="$BASE/gnn_refs"
CHECKPOINTS="$BASE/v19_gnn_marble_validation"
OUTPUT="$BASE/v19_gnn_marble_final_test"
CACHE="$BASE/sft_models/v19_gnn_component_cache_v3"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

cd "$REPO"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to reuse an existing final-test directory: $OUTPUT" >&2
  exit 2
fi

python "$PKG/server_scripts/v19_component_gnn_multitask.py" final-test \
  --checkpoint-dir "$CHECKPOINTS/gsafeguard_gat_v2_zero_truncation" --official-dir "$REFS/G-safeguard/TA" \
  --test-file "$DATA/test.jsonl" --cache-dir "$CACHE" \
  --output-dir "$OUTPUT/gsafeguard_gat_v2_zero_truncation" --sealed-test-ack FINAL_ONCE

python "$PKG/server_scripts/v19_unsupervised_graph_baselines.py" final-test \
  --checkpoint-dir "$CHECKPOINTS/blindguard_scl_v19" --official-dir "$REFS/BlindGuard/MA" \
  --data-dir "$DATA" --cache-dir "$CACHE/bilevel" \
  --output-dir "$OUTPUT/blindguard_scl_v19" --sealed-test-ack FINAL_ONCE

python "$PKG/server_scripts/v19_unsupervised_graph_baselines.py" final-test \
  --checkpoint-dir "$CHECKPOINTS/xgguard_bilevel_v19" --official-dir "$REFS/XG-Guard" \
  --data-dir "$DATA" --cache-dir "$CACHE/bilevel" \
  --output-dir "$OUTPUT/xgguard_bilevel_v19" --sealed-test-ack FINAL_ONCE

python "$V19/scripts/write_sha256_manifest.py" "$OUTPUT"
tar -czf "$BASE/v19_gnn_marble_final_test.tar.gz" -C "$BASE" "$(basename "$OUTPUT")"
echo "Sealed GNN final test complete: $BASE/v19_gnn_marble_final_test.tar.gz"
