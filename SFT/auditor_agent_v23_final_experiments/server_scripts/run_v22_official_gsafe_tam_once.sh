#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"
SUP="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}"
REFS="${BASELINE_REFS:-$BASE/baseline_repos}"
SCRIPT="$REPO/SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_component_gnn_multitask.py"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR to v23_final_aligned_combined}"; TEST="$DATA/test.jsonl"; GPU="${GPU:-0}"
GSAFE="$REFS/G-safeguard"; BLIND="$REFS/BlindGuard"
export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}" TOKENIZERS_PARALLELISM=false
mkdir -p "$REFS" "$SUP/baselines"
if [[ ! -d "$GSAFE/.git" ]]; then git clone https://github.com/wslong20/G-safeguard.git "$GSAFE"; fi
if [[ ! -d "$BLIND/.git" ]]; then git clone https://github.com/MR9812/BlindGuard.git "$BLIND"; fi
git -C "$GSAFE" cat-file -e '890c99f1cbc864e9ff0c85859619a14f42bc9cab^{commit}' 2>/dev/null || git -C "$GSAFE" fetch origin
git -C "$GSAFE" checkout --detach 890c99f1cbc864e9ff0c85859619a14f42bc9cab
git -C "$BLIND" cat-file -e '1889c20a326ba9ba9a6982744d473626e74f9986^{commit}' 2>/dev/null || git -C "$BLIND" fetch origin
git -C "$BLIND" checkout --detach 1889c20a326ba9ba9a6982744d473626e74f9986
train_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$DATA/train.jsonl','rb').read()).hexdigest())")"
val_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$DATA/validation.jsonl','rb').read()).hexdigest())")"
test_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$TEST','rb').read()).hexdigest())")"

run_method() {
  local kind="$1" official="$2" name="$3" out="$SUP/baselines/${3}_official_v23_v1"
  local cache="$SUP/baselines/v23_official_gnn_shared_component_cache_v1"
  mkdir -p "$cache"
  if [[ ! -f "$out/smoke_model/TRAIN_CONTRACT.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind "$kind" --official-dir "$official" \
      --data-dir "$DATA" --cache-dir "$cache" --output-dir "$out/smoke_model" --epochs 1 --limit 100 \
      --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha"
  fi
  if [[ ! -f "$out/smoke_test/metrics.json" ]]; then
    rm -rf -- "$out/smoke_test.incomplete"
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$out/smoke_model" --official-dir "$official" \
      --test-file "$DATA/validation.jsonl" --cache-dir "$cache" --output-dir "$out/smoke_test.incomplete" \
      --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$val_sha" --limit 100
    rm -rf -- "$out/smoke_test"
    mv "$out/smoke_test.incomplete" "$out/smoke_test"
  fi
  python - "$out/smoke_test/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x["n"]==100
print("OFFICIAL_ENCODER_V23_SMOKE: PASS",x["method"])
PY
  if [[ ! -f "$out/model/TRAIN_CONTRACT.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" train-validation --model-kind "$kind" --official-dir "$official" \
      --data-dir "$DATA" --cache-dir "$cache" --output-dir "$out/model" \
      --epochs "${GNN_OFFICIAL_EPOCHS:-20}" --grad-accum "${GNN_GRAD_ACCUM:-16}" \
      --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha"
  fi
  if [[ ! -f "$out/test/metrics.json" ]]; then
    rm -rf -- "$out/test.incomplete"
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT" final-test --checkpoint-dir "$out/model" --official-dir "$official" \
      --test-file "$TEST" --cache-dir "$cache" --output-dir "$out/test.incomplete" \
      --sealed-test-ack FINAL_ONCE --expected-test-sha256 "$test_sha"
    rm -rf -- "$out/test"
    mv "$out/test.incomplete" "$out/test"
  fi
  python - "$out/model/TRAIN_CONTRACT.json" "$out/test/metrics.json" "$kind" "$test_sha" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]));kind=sys.argv[3];test_sha=sys.argv[4]
assert c['model_kind']==kind and c['test_accessed'] is False
assert m['model_kind']==kind and m['dataset_role']=='test' and m['data_sha256']==test_sha and m['n']==6207
assert all(k in m for k in ('three_class_accuracy','three_class_report','binary_accuracy','localization'))
print({'status':'PASS','model_kind':kind,'test_rows':m['n']})
PY
  echo "DONE: $out/test/metrics.json"
}
methods=",${GNN_METHODS:-gat,tam},"
[[ "$methods" != *,gat,* ]] || run_method gat "$GSAFE/MA" gsafeguard
[[ "$methods" != *,tam,* ]] || run_method tam "$BLIND/MA" tam
