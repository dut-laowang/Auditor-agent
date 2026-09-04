#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
LEGACY_RUN="${V23_RUN:-$BASE/v23_final_run}"
OUT="${V23_EXPERIMENT_RUN:-$BASE/v23_final_experiment_run}"
GPU="${GPU:-0}"
METHODS="${METHODS:-modernbert}"
PKG="$REPO/SFT/auditor_agent_v23_final_experiments"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR to v23_final_aligned_combined}"
FOLDS="$OUT/heldout_data"
MODEL_REV="b968826d9c46dd6066d109eabc6255188de91218"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
cd "$REPO"

[[ "$(wc -l < "$DATA/train.jsonl" | tr -d ' ')" -eq 30619 ]]
[[ "$(wc -l < "$DATA/validation.jsonl" | tr -d ' ')" -eq 7018 ]]
python "$PKG/scripts/build_heldout_splits.py" --data-dir "$DATA" --output-dir "$FOLDS" --modernbert-zero-truncation

for spec in topology__tree surface__message scenario__research; do
  fold="$FOLDS/$spec"
  train_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$fold/train.jsonl','rb').read()).hexdigest())")"
  val_sha="$(python -c "import hashlib;print(hashlib.sha256(open('$fold/validation.jsonl','rb').read()).hexdigest())")"
  if [[ ",$METHODS," == *,modernbert,* ]]; then
    # v2 is deliberately isolated from the pre-filter v1 contract. The first
    # failed preflight may already have written CONTRACT.json even though no
    # checkpoint exists; reusing that directory would correctly be rejected.
    model="$OUT/heldout/$spec/modernbert_model_ztr_v2"
    result="$OUT/heldout/$spec/modernbert_ztr_v2"
    modern_epochs="${MODERN_EPOCHS:-3}"
    checkpoint="$model/checkpoint-epoch-${modern_epochs}.pt"
    if [[ ! -f "$checkpoint" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
        --mode train --data-file "$fold/train.jsonl" --dataset-role train --output-dir "$model" \
        --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha" \
        --expected-test-sha256 UNUSED --epochs "$modern_epochs" \
        --batch "${MODERN_TRAIN_BATCH:-2}" --grad-accum "${MODERN_GRAD_ACCUM:-8}" \
        2>&1 | tee -a "$OUT/logs/${spec}_modernbert_train.log"
    fi
    CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
      --mode eval --data-file "$fold/validation.jsonl" --dataset-role validation \
      --checkpoint "$checkpoint" --output-dir "$result" \
      --expected-train-sha256 "$train_sha" --expected-validation-sha256 "$val_sha" \
      --expected-test-sha256 UNUSED --batch "${MODERN_EVAL_BATCH:-4}" \
      2>&1 | tee -a "$OUT/logs/${spec}_modernbert_eval.log"
  fi
  if [[ ",$METHODS," == *,qwen,* ]]; then
    model="$OUT/heldout/$spec/qwen_model"
    result="$OUT/heldout/$spec/qwen"
    if [[ ! -f "$model/adapter_model.safetensors" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
        --model Qwen/Qwen3-8B --revision "$MODEL_REV" --data-dir "$fold" --output-dir "$model" \
        --max-len 12288 --context-contract v23-all-12288 --prompt-overflow error \
        --epochs "${QWEN_EPOCHS:-2}" --batch "${QWEN_TRAIN_BATCH:-1}" \
        --grad-accum "${QWEN_GRAD_ACCUM:-16}" --resume auto --disable-cudnn-sdp \
        2>&1 | tee -a "$OUT/logs/${spec}_qwen_train.log"
    fi
    CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
      --mode sft --model Qwen/Qwen3-8B --revision "$MODEL_REV" --adapter "$model" \
      --test-file "$fold/validation.jsonl" --dataset-role validation --output-dir "$result" \
      --max-input-len 12288 --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" \
      --resume --disable-cudnn-sdp 2>&1 | tee -a "$OUT/logs/${spec}_qwen_eval.log"
  fi
done

python "$PKG/scripts/render_experiment_tables.py" --run-dir "$LEGACY_RUN" --supplement-dir "$OUT" --output-dir "$OUT/tables"
python - "$OUT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=root/'heldout_data/HELDOUT_MANIFEST.json'
done={"version":"V23-final-heldout-suite-v1","status":"PASS","heldout_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}
(root/'SUPPLEMENT_SUITE_COMPLETE.json').write_text(json.dumps(done,indent=2),encoding='utf8')
print(json.dumps(done,indent=2))
PY
