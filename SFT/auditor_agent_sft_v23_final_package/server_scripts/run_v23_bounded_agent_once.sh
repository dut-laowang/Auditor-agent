#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}";REPO="${REPO:-$BASE/Auditor-agent}";RUN="${V23_RUN:-$BASE/v23_final_run}";DATA="${V23_DATA_DIR:?Set V23_DATA_DIR}";GPU="${GPU:-0}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package";V22="$REPO/SFT/auditor_agent_sft_v22_all_package";OUT="$RUN/bounded_agent_common6167";QMODEL="$RUN/models/qwen3_8b_plain_sft"
for p in "$RUN/qwen3_8b_plain_sft_validation/predictions.jsonl" "$RUN/modernbert_eval/predictions.jsonl" "$RUN/qwen3_8b_plain_sft_test/predictions.jsonl" "$RUN/modernbert_test/predictions.jsonl";do test -f "$p";done
mkdir -p "$OUT"
python "$V22/scripts/v22_plain_hetero_agent.py" prepare --validation-qwen "$RUN/qwen3_8b_plain_sft_validation/predictions.jsonl" --validation-bert "$RUN/modernbert_eval/predictions.jsonl" --validation-index "$DATA/validation_track_index.jsonl" --test-data "$DATA/test.jsonl" --test-index "$DATA/test_track_index.jsonl" --test-qwen "$RUN/qwen3_8b_plain_sft_test/predictions.jsonl" --test-bert "$RUN/modernbert_test/predictions.jsonl" --output-dir "$OUT" --rows 6167 --max-verify-rate "${AGENT_MAX_VERIFY_RATE:-0.15}"
if [[ -s "$OUT/rewrite_data.jsonl" ]];then CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --adapter "$QMODEL" --test-file "$OUT/rewrite_data.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE --output-dir "$OUT/rewrite_eval" --max-input-len 12288 --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp --structured-controls "$OUT/rewrite_controls.jsonl";else mkdir -p "$OUT/rewrite_eval";:>"$OUT/rewrite_eval/predictions.jsonl";fi
python "$V22/scripts/v22_plain_hetero_agent.py" merge --base-predictions "$OUT/base_predictions.jsonl" --decisions "$OUT/agent_decisions.jsonl" --rewrite-predictions "$OUT/rewrite_eval/predictions.jsonl" --output-dir "$OUT"
python "$V22/scripts/score_predictions_by_track.py" --predictions "$OUT/agent_predictions.jsonl" --track-index "$OUT/test_subset_index.jsonl" --output-dir "$OUT/by_track"
python - "$OUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads((p/'AGENT_TEST_COMPARISON.json').read_text());assert r['rows']==6167 and r['verify_rate']<=.150000001
(p/'V23_AGENT_COMPLETE.json').write_text(json.dumps({'status':'PASS','rows':6167,'verify_rate':r['verify_rate']},indent=2))
PY
