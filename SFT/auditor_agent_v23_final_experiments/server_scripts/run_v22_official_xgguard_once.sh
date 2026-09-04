#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"
EXP="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR}"
OUT="$EXP/baselines/xgguard_official_v23_v1"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"
OFFICIAL="$REFS/XG-Guard"
SCRIPT="$REPO/SFT/auditor_agent_v23_final_experiments/scripts/v22_xgguard_official_adapter.py"
TRAIN="$DATA/train.jsonl"
VAL="$DATA/validation.jsonl"
TEST="$DATA/test.jsonl"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REFS" "$(dirname "$OUT")" "$OUT/cache"
for f in "$TRAIN" "$VAL" "$TEST" "$SCRIPT"; do [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }; done
if [[ ! -d "$OFFICIAL/.git" ]]; then git clone https://github.com/CampanulaBells/XG-Guard.git "$OFFICIAL"; fi
git -C "$OFFICIAL" fetch origin
git -C "$OFFICIAL" checkout --detach 86e1121512f76800f80d4687e492c7f99f049929

# Gate 1: official-source identity + schema/model smoke. Test is deliberately
# validation here, so the sealed test is not touched by the smoke run.
if [[ ! -f "$OUT/smoke/RUN_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" --official-dir "$OFFICIAL" \
    --train "$TRAIN" --validation "$VAL" --test "$VAL" \
    --cache-dir "$OUT/cache" --output-dir "$OUT/smoke" --smoke-only
fi
python - "$OUT/smoke/RUN_COMPLETE.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x["status"]=="PASS" and x["smoke_only"] is True
print("OFFICIAL_XGGUARD_SMOKE: PASS")
PY

# Gate 2: full normal-only training, validation-only calibration, one final test.
if [[ ! -f "$OUT/full/RUN_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" --official-dir "$OFFICIAL" \
    --train "$TRAIN" --validation "$VAL" --test "$TEST" \
    --cache-dir "$OUT/cache" --output-dir "$OUT/full" \
    --epochs "${XGGUARD_OFFICIAL_EPOCHS:-20}" --batch-size "${XGGUARD_OFFICIAL_BATCH:-8}"
fi
python - "$OUT/full/RUN_COMPLETE.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x["status"]=="PASS" and x["smoke_only"] is False
print(json.dumps(x,indent=2))
PY
echo "DONE: $OUT/full/metrics.json"
