#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"; REPO="${REPO:-$BASE/Auditor-agent}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR}"; OUT="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}/baselines/blindguard_official_v23_v1"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"; OFFICIAL="$REFS/BlindGuard"; GPU="${GPU:-0}"
SCRIPT="$REPO/SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_unsupervised_graph_baselines.py"
CACHE="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}/baselines/v23_blindguard_component_cache"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}" TOKENIZERS_PARALLELISM=false
mkdir -p "$REFS" "$(dirname "$OUT")" "$CACHE"
[[ -f "$SCRIPT" ]] || { echo "MISSING: $SCRIPT" >&2; exit 2; }
if [[ ! -d "$OFFICIAL/.git" ]]; then git clone https://github.com/MR9812/BlindGuard.git "$OFFICIAL"; fi
git -C "$OFFICIAL" cat-file -e '1889c20a326ba9ba9a6982744d473626e74f9986^{commit}' 2>/dev/null || git -C "$OFFICIAL" fetch origin
git -C "$OFFICIAL" checkout --detach 1889c20a326ba9ba9a6982744d473626e74f9986
sha(){ sha256sum "$1" | awk '{print $1}'; }
TRAIN_SHA="$(sha "$DATA/train.jsonl")"; VAL_SHA="$(sha "$DATA/validation.jsonl")"; TEST_SHA="$(sha "$DATA/test.jsonl")"

# Faithful BlindGuard SCL: official GATSCL, clean graphs only, synthetic node
# corruption, no attack labels in optimization. V23 labels are scoring-only.
if [[ ! -f "$OUT/model/TRAIN_CONTRACT.json" ]]; then
 CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind blindguard \
  --official-dir "$OFFICIAL/MA" --data-dir "$DATA" --cache-dir "$CACHE" --output-dir "$OUT/model" \
  --epochs "${BLINDGUARD_EPOCHS:-10}" --batch-size 1 --lr 0.001 --weight-decay 0.0002 \
  --hidden-dim 1024 --latent-dim 512 --seed 3701 \
  --expected-train-sha256 "$TRAIN_SHA" --expected-validation-sha256 "$VAL_SHA"
fi
if [[ ! -f "$OUT/test/metrics.json" ]]; then
 rm -rf -- "$OUT/test" "$OUT/test.incomplete"
 CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$OUT/model" \
  --official-dir "$OFFICIAL/MA" --data-dir "$DATA" --cache-dir "$CACHE" --output-dir "$OUT/test.incomplete" \
  --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$TEST_SHA"
 mv "$OUT/test.incomplete" "$OUT/test"
fi
python - "$OUT/model/TRAIN_CONTRACT.json" "$OUT/test/metrics.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]))
assert c['model_kind']=='blindguard' and c['normal_only_training_rows']>0 and m['model_kind']=='blindguard' and m['n']==6207
assert all(k in m for k in ('three_class_accuracy','three_class_report','binary_accuracy','localization'))
assert c['official_commit']=='1889c20a326ba9ba9a6982744d473626e74f9986'
print({'status':'PASS','method':'BlindGuard SCL','normal_only_rows':c['normal_only_training_rows'],'test_rows':m['n']})
PY
