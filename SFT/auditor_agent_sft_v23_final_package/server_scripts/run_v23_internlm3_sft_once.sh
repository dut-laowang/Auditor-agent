#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR to v23_final_aligned_combined}"
GPU="${GPU:-0}"
MODEL_ID="internlm/internlm3-8b-instruct"
MODEL_REVISION="${INTERNLM_REVISION:-28c9941}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
V22="$REPO/SFT/auditor_agent_sft_v22_all_package"
V23="$REPO/SFT/auditor_agent_sft_v23_final_package"
MODEL="$RUN/models/internlm3_8b_sft"
VAL="$RUN/internlm3_8b_sft_validation"
TEST="$RUN/internlm3_8b_sft_test"
AUDIT="$RUN/data_contract/internlm3_context_audit.json"
LOGS="$RUN/logs"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$REPO"; mkdir -p "$RUN/data_contract" "$RUN/models" "$LOGS"

# InternLM has a different tokenizer. It must pass its own complete no-truncate
# gate before training; Qwen token counts are not reused.
if [[ ! -f "$AUDIT" ]]; then
  python "$V23/scripts/audit_v23_qwen_sft_contract.py" \
    --data-dir "$DATA" --output "$AUDIT" --model "$MODEL_ID" \
    --revision "$MODEL_REVISION" --max-len 12288 --batch-size 64
fi
python - "$AUDIT" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("status") != "PASS" or r.get("total_issues") != 0:
    raise RuntimeError("InternLM context/SFT contract did not pass")
if sum(x["rows"] for x in r["splits"].values()) != 43844:
    raise RuntimeError("InternLM context audit did not cover all V23 rows")
PY

if [[ ! -f "$MODEL/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model "$MODEL_ID" --revision "$MODEL_REVISION" --data-dir "$DATA" --output-dir "$MODEL" \
    --max-len 12288 --context-contract v23-all-12288 --prompt-overflow error \
    --epochs "${INTERNLM_EPOCHS:-2}" --lr "${INTERNLM_LR:-2e-4}" \
    --batch "${INTERNLM_TRAIN_BATCH:-1}" --grad-accum "${INTERNLM_GRAD_ACCUM:-16}" \
    --seed 42 --resume auto --disable-cudnn-sdp \
    2>&1 | tee -a "$LOGS/internlm3_8b_sft_training.log"
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model "$MODEL_ID" --revision "$MODEL_REVISION" --adapter "$MODEL" \
  --test-file "$DATA/validation.jsonl" --dataset-role validation --output-dir "$VAL" \
  --max-input-len 12288 --max-new-tokens 1400 --batch-size "${INTERNLM_EVAL_BATCH:-4}" \
  --resume --disable-cudnn-sdp 2>&1 | tee -a "$LOGS/internlm3_validation.log"
python "$V22/scripts/score_predictions_by_track.py" --predictions "$VAL/predictions.jsonl" \
  --track-index "$DATA/validation_track_index.jsonl" --output-dir "$VAL/by_track"

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model "$MODEL_ID" --revision "$MODEL_REVISION" --adapter "$MODEL" \
  --test-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE --output-dir "$TEST" \
  --max-input-len 12288 --max-new-tokens 1400 --batch-size "${INTERNLM_EVAL_BATCH:-4}" \
  --resume --disable-cudnn-sdp 2>&1 | tee -a "$LOGS/internlm3_test.log"
python "$V22/scripts/score_predictions_by_track.py" --predictions "$TEST/predictions.jsonl" \
  --track-index "$DATA/test_track_index.jsonl" --output-dir "$TEST/by_track"

python - "$MODEL/run_manifest.json" "$VAL/by_track/metrics_by_track.json" "$TEST/by_track/metrics_by_track.json" "$RUN" <<'PY'
import hashlib,json,pathlib,sys
model,valp,testp,run=map(pathlib.Path,sys.argv[1:]); m=json.loads(model.read_text());v=json.loads(valp.read_text());t=json.loads(testp.read_text())
if m.get("version")!="V23-ALL-audit-grade-sft-v1" or m.get("context_contract")!="v23-all-12288":raise RuntimeError("InternLM model contract mismatch")
if v["all"]["n"]!=7018 or t["all"]["n"]!=6207:raise RuntimeError("InternLM evaluation incomplete")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={"version":"V23-InternLM3-8B-SFT-v1","status":"PASS","model_revision":m["model_revision"],"validation_rows":7018,"test_rows":6207,"model_manifest_sha256":sha(model),"validation_metrics_sha256":sha(valp),"test_metrics_sha256":sha(testp)}
(run/"INTERNLM3_SFT_COMPLETE.json").write_text(json.dumps(out,indent=2),encoding="utf-8");print(json.dumps(out,indent=2))
PY
