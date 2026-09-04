#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"; REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"; DATA="${V23_DATA_DIR:?Set V23_DATA_DIR}"
GPU="${GPU:-0}"; V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"; V22="$REPO/SFT/auditor_agent_sft_v22_all_package"
FILTER="$V22/scripts/filter_v22_expanded_modernbert_context.py"; OUT="$RUN/modernbert_data"; MODEL="$RUN/models/modernbert"; VAL="$RUN/modernbert_eval"; TEST="$RUN/modernbert_test"; LOG="$RUN/logs"
mkdir -p "$LOG" "$RUN/models"
if [[ ! -f "$OUT/MODERNBERT_CONTEXT_GATE.json" ]];then python "$FILTER" --data-dir "$DATA" --index-dir "$DATA" --output-dir "$OUT" --splits train validation test --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --max-len 8192;fi
[[ "$(wc -l < "$OUT/train.jsonl"|tr -d ' ')" -eq 30549 ]];[[ "$(wc -l < "$OUT/validation.jsonl"|tr -d ' ')" -eq 6969 ]];[[ "$(wc -l < "$OUT/test.jsonl"|tr -d ' ')" -eq 6167 ]]
sha(){ sha256sum "$1"|awk '{print $1}';}; TS="$(sha "$OUT/train.jsonl")"; VS="$(sha "$OUT/validation.jsonl")"; XS="$(sha "$OUT/test.jsonl")"
if [[ ! -f "$MODEL/TRAINING_COMPLETE.json" ]];then CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" --mode train --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --data-file "$OUT/train.jsonl" --dataset-role train --output-dir "$MODEL" --expected-train-sha256 "$TS" --expected-validation-sha256 "$VS" --expected-test-sha256 "$XS" --max-len 8192 --attn-implementation sdpa --input-mode user --epochs "${MODERN_EPOCHS:-3}" --lr "${MODERN_LR:-2e-5}" --batch "${MODERN_TRAIN_BATCH:-2}" --grad-accum "${MODERN_GRAD_ACCUM:-8}" --seed 42 2>&1|tee -a "$LOG/modernbert_train.log";fi
CKPT="$MODEL/checkpoint-epoch-${MODERN_EPOCHS:-3}.pt"
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --checkpoint "$CKPT" --data-file "$OUT/validation.jsonl" --dataset-role validation --output-dir "$VAL" --expected-train-sha256 "$TS" --expected-validation-sha256 "$VS" --expected-test-sha256 "$XS" --max-len 8192 --attn-implementation sdpa --input-mode user --batch "${MODERN_EVAL_BATCH:-4}" --seed 42
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --checkpoint "$CKPT" --data-file "$OUT/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE --output-dir "$TEST" --expected-train-sha256 "$TS" --expected-validation-sha256 "$VS" --expected-test-sha256 "$XS" --max-len 8192 --attn-implementation sdpa --input-mode user --batch "${MODERN_EVAL_BATCH:-4}" --seed 42
python "$V22/scripts/score_predictions_by_track.py" --predictions "$TEST/predictions.jsonl" --track-index "$OUT/test_track_index.jsonl" --output-dir "$TEST/by_track"
python - "$TEST/metrics.json" "$RUN" "$DATA/test.jsonl" "$OUT/test.jsonl" <<'PY'
import hashlib,json,pathlib,sys
metric,run,source_test,filtered_test=map(pathlib.Path,sys.argv[1:]);m=json.loads(metric.read_text());assert m['n']==6167
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(run/'MODERNBERT_COMPLETE.json').write_text(json.dumps({'status':'PASS','rows':6167,'source_test_sha256':sha(source_test),'filtered_test_sha256':sha(filtered_test),'metrics_sha256':sha(metric)},indent=2))
PY
