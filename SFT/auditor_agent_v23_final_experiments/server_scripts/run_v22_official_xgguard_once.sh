#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"
EXP="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR}"
OUT="$EXP/baselines/xgguard_official_v23_v2"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"
OFFICIAL="$REFS/XG-Guard"
SCRIPT="$REPO/SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_unsupervised_graph_baselines.py"
TRAIN="$DATA/train.jsonl"
VAL="$DATA/validation.jsonl"
TEST="$DATA/test.jsonl"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REFS" "$(dirname "$OUT")" "$OUT/cache"
for f in "$TRAIN" "$VAL" "$TEST" "$SCRIPT"; do [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }; done
if [[ ! -d "$OFFICIAL/.git" ]]; then git clone https://github.com/CampanulaBells/XG-Guard.git "$OFFICIAL"; fi
git -C "$OFFICIAL" cat-file -e '86e1121512f76800f80d4687e492c7f99f049929^{commit}' 2>/dev/null || git -C "$OFFICIAL" fetch origin
git -C "$OFFICIAL" checkout --detach 86e1121512f76800f80d4687e492c7f99f049929

sha(){ sha256sum "$1" | awk '{print $1}'; }
TRAIN_SHA="$(sha "$TRAIN")"; VAL_SHA="$(sha "$VAL")"; TEST_SHA="$(sha "$TEST")"

# Two-phase protocol: normal-only official objective and validation calibration
# first; the sealed test file is not opened until the checkpoint is complete.
if [[ ! -f "$OUT/model/TRAIN_CONTRACT.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind xgguard \
    --official-dir "$OFFICIAL" --data-dir "$DATA" --cache-dir "$OUT/cache" --output-dir "$OUT/model" \
    --epochs "${XGGUARD_OFFICIAL_EPOCHS:-20}" --batch-size "${XGGUARD_OFFICIAL_BATCH:-8}" \
    --lr "${XGGUARD_OFFICIAL_LR:-1e-4}" --weight-decay "${XGGUARD_OFFICIAL_WEIGHT_DECAY:-2e-4}" \
    --alpha "${XGGUARD_OFFICIAL_ALPHA:-1e-4}" --seed 3701 \
    --expected-train-sha256 "$TRAIN_SHA" --expected-validation-sha256 "$VAL_SHA"
fi
if [[ ! -f "$OUT/test/metrics.json" ]]; then
  rm -rf -- "$OUT/test.incomplete"
  CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$OUT/model" \
    --official-dir "$OFFICIAL" --data-dir "$DATA" --cache-dir "$OUT/cache" --output-dir "$OUT/test.incomplete" \
    --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$TEST_SHA"
  rm -rf -- "$OUT/test"
  mv "$OUT/test.incomplete" "$OUT/test"
fi
python - "$OUT/model/TRAIN_CONTRACT.json" "$OUT/test/metrics.json" "$TEST_SHA" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]));test_sha=sys.argv[3]
assert c['model_kind']=='xgguard' and c['normal_only_training_rows']>0 and c['test_accessed'] is False
assert m['model_kind']=='xgguard' and m['dataset_role']=='test' and m['data_sha256']==test_sha and m['n']==6207
assert c['official_commit']=='86e1121512f76800f80d4687e492c7f99f049929'
assert all(k in m for k in ('three_class_accuracy','three_class_report','binary_accuracy','localization'))
print({'status':'PASS','method':'XG-Guard','normal_only_rows':c['normal_only_training_rows'],'test_rows':m['n']})
PY
echo "DONE: $OUT/test/metrics.json"
