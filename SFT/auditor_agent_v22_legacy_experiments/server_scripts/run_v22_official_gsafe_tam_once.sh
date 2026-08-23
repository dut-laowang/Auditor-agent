#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
SUP="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"
SCRIPT="$REPO/SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_component_gnn_multitask.py"
DATA="$RUN/base_dataset"; TEST="$RUN/modernbert_sealed_test_source/test.jsonl"; GPU="${GPU:-0}"
GSAFE="$REFS/G-safeguard"; BLIND="$REFS/BlindGuard"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}" TOKENIZERS_PARALLELISM=false
mkdir -p "$REFS" "$SUP/baselines"
if [[ ! -d "$GSAFE/.git" ]]; then git clone https://github.com/wslong20/G-safeguard.git "$GSAFE"; fi
if [[ ! -d "$BLIND/.git" ]]; then git clone https://github.com/MR9812/BlindGuard.git "$BLIND"; fi
git -C "$GSAFE" fetch origin; git -C "$GSAFE" checkout --detach 890c99f1cbc864e9ff0c85859619a14f42bc9cab
git -C "$BLIND" fetch origin; git -C "$BLIND" checkout --detach 1889c20a326ba9ba9a6982744d473626e74f9986
train_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$DATA/train.jsonl','rb').read()).hexdigest())")"
val_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$DATA/validation.jsonl','rb').read()).hexdigest())")"
test_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$TEST','rb').read()).hexdigest())")"

run_method() {
  local kind="$1" official="$2" name="$3" out="$SUP/baselines/${3}_official_v22_v1"
  local cache="$SUP/baselines/v22_official_gnn_shared_component_cache_v1"
  mkdir -p "$cache"
  if [[ ! -f "$out/smoke_model/TRAIN_CONTRACT.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind "$kind" --official-dir "$official" \
      --data-dir "$DATA" --cache-dir "$cache" --output-dir "$out/smoke_model" --epochs 1 --limit 100 \
      --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha"
  fi
  if [[ ! -f "$out/smoke_test/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$out/smoke_model" --official-dir "$official" \
      --test-file "$DATA/validation.jsonl" --cache-dir "$cache" --output-dir "$out/smoke_test" \
      --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$val_sha" --limit 100
  fi
  python - "$out/smoke_test/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x["n"]==100
print("OFFICIAL_ENCODER_V22_SMOKE: PASS",x["method"])
PY
  if [[ ! -f "$out/model/TRAIN_CONTRACT.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind "$kind" --official-dir "$official" \
      --data-dir "$DATA" --cache-dir "$cache" --output-dir "$out/model" \
      --epochs "${GNN_OFFICIAL_EPOCHS:-20}" --grad-accum "${GNN_GRAD_ACCUM:-16}" \
      --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha"
  fi
  if [[ ! -f "$out/test/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$out/model" --official-dir "$official" \
      --test-file "$TEST" --cache-dir "$cache" --output-dir "$out/test" \
      --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$test_sha"
  fi
  echo "DONE: $out/test/metrics.json"
}
methods=",${GNN_METHODS:-gat,tam},"
[[ "$methods" != *,gat,* ]] || run_method gat "$GSAFE/MA" gsafeguard
[[ "$methods" != *,tam,* ]] || run_method tam "$BLIND/MA" tam
