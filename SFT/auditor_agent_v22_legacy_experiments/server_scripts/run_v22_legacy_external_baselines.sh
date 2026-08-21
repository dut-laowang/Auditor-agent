#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"
GPU="${GPU:-0}"
GNN="$REPO/SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_unsupervised_graph_baselines.py"
mkdir -p "$REFS" "$OUT/baselines" "$OUT/cache"

if [[ ! -d "$REFS/XG-Guard/.git" ]]; then git clone https://github.com/CampanulaBells/XG-Guard.git "$REFS/XG-Guard"; fi
git -C "$REFS/XG-Guard" fetch origin
git -C "$REFS/XG-Guard" checkout --detach 86e1121512f76800f80d4687e492c7f99f049929
train_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$RUN/base_dataset/train.jsonl','rb').read()).hexdigest())")"
val_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$RUN/base_dataset/validation.jsonl','rb').read()).hexdigest())")"
if [[ ! -f "$OUT/baselines/xgguard/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$GNN" train-validation \
    --model-kind xgguard --official-dir "$REFS/XG-Guard" --data-dir "$RUN/base_dataset" \
    --cache-dir "$OUT/cache/xgguard" --output-dir "$OUT/baselines/xgguard" \
    --epochs "${XGGUARD_EPOCHS:-50}" --batch-size "${XGGUARD_BATCH:-8}" \
    --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha"
fi

if [[ "${RUN_XGGUARD_TEST:-0}" == 1 && ! -f "$OUT/baselines/xgguard_test/metrics.json" ]]; then
  XGDATA="$OUT/xgguard_data"
  mkdir -p "$XGDATA"
  ln -sfn "$RUN/base_dataset/train.jsonl" "$XGDATA/train.jsonl"
  ln -sfn "$RUN/base_dataset/validation.jsonl" "$XGDATA/validation.jsonl"
  ln -sfn "$RUN/modernbert_sealed_test_source/test.jsonl" "$XGDATA/test.jsonl"
  test_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$XGDATA/test.jsonl','rb').read()).hexdigest())")"
  CUDA_VISIBLE_DEVICES="$GPU" python "$GNN" final-test \
    --checkpoint-dir "$OUT/baselines/xgguard" --official-dir "$REFS/XG-Guard" \
    --data-dir "$XGDATA" --cache-dir "$OUT/cache/xgguard" \
    --output-dir "$OUT/baselines/xgguard_test" --sealed-test-ack FINAL_ONCE \
    --expected-test-sha256 "$test_sha"
fi

# AgentForesight is evaluated only under its native online prefix protocol.
# It is deliberately not converted into a post-hoc row in our main table.
if [[ "${RUN_AGENTFORESIGHT_NATIVE:-0}" == 1 ]]; then
  if [[ ! -d "$REFS/AgentForesight/.git" ]]; then git clone https://github.com/ZBox1005/AgentForesight.git "$REFS/AgentForesight"; fi
  git -C "$REFS/AgentForesight" fetch origin
  git -C "$REFS/AgentForesight" checkout --detach af549083fcc6c3eff4ccc558534515e46ca0578e
  python -m pip install -r "$REFS/AgentForesight/requirements.txt"
  (cd "$REFS/AgentForesight" && CUDA_VISIBLE_DEVICES="$GPU" python -m inference.infer_local \
    --model-path "${AGENTFORESIGHT_MODEL:?set AGENTFORESIGHT_MODEL}" \
    --data-dir "${AGENTFORESIGHT_DATA:?set AGENTFORESIGHT_DATA}" \
    --output-dir "$OUT/baselines/agentforesight_native" --paper-test-split)
fi
