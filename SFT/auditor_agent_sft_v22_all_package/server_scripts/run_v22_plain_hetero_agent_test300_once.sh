#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
GPU="${GPU:-0}"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
OUT="$RUN/plain_hetero_agent_test300"
TEST_SOURCE="$RUN/modernbert_sealed_test_source"
PLAIN_MODEL="$RUN/models/qwen3_8b_plain_sft"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
for path in \
  "$RUN/base_dataset/validation.jsonl" "$RUN/track_index/validation.jsonl" \
  "$RUN/qwen3_8b_plain_sft_validation/predictions.jsonl" "$RUN/modernbert_eval/predictions.jsonl" \
  "$TEST_SOURCE/test.jsonl" "$TEST_SOURCE/track_index.jsonl" \
  "$RUN/qwen3_8b_plain_sft_test/predictions.jsonl" "$RUN/modernbert_sealed_test/predictions.jsonl" \
  "$PLAIN_MODEL/run_manifest.json"; do test -f "$path"; done
mkdir -p "$OUT"

python "$ALL/scripts/v22_plain_hetero_agent.py" prepare \
  --validation-qwen "$RUN/qwen3_8b_plain_sft_validation/predictions.jsonl" \
  --validation-bert "$RUN/modernbert_eval/predictions.jsonl" \
  --validation-index "$RUN/track_index/validation.jsonl" \
  --test-data "$TEST_SOURCE/test.jsonl" --test-index "$TEST_SOURCE/track_index.jsonl" \
  --test-qwen "$RUN/qwen3_8b_plain_sft_test/predictions.jsonl" \
  --test-bert "$RUN/modernbert_sealed_test/predictions.jsonl" \
  --output-dir "$OUT" --rows "${AGENT_TEST_ROWS:-300}" \
  --max-verify-rate "${AGENT_MAX_VERIFY_RATE:-0.15}"

if [[ -s "$OUT/rewrite_data.jsonl" ]]; then
  if [[ -f "$OUT/rewrite_eval/predictions.jsonl" && ! -s "$OUT/rewrite_eval/predictions.jsonl" \
      && ! -f "$OUT/rewrite_eval/EVAL_CONTRACT.json" ]]; then rm -- "$OUT/rewrite_eval/predictions.jsonl"; fi
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
    --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --adapter "$PLAIN_MODEL" --test-file "$OUT/rewrite_data.jsonl" --dataset-role test \
    --sealed-test-ack FINAL_ONCE --output-dir "$OUT/rewrite_eval" --max-input-len 12288 \
    --max-new-tokens 1400 --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp \
    --structured-controls "$OUT/rewrite_controls.jsonl"
else
  mkdir -p "$OUT/rewrite_eval"; : > "$OUT/rewrite_eval/predictions.jsonl"
fi

python "$ALL/scripts/v22_plain_hetero_agent.py" merge \
  --base-predictions "$OUT/base_predictions.jsonl" --decisions "$OUT/agent_decisions.jsonl" \
  --rewrite-predictions "$OUT/rewrite_eval/predictions.jsonl" --output-dir "$OUT"
python "$ALL/scripts/score_predictions_by_track.py" --predictions "$OUT/agent_predictions.jsonl" \
  --track-index "$OUT/test_300_index.jsonl" --output-dir "$OUT/by_track"
python - "$OUT" "${AGENT_TEST_ROWS:-300}" "${AGENT_MAX_VERIFY_RATE:-0.15}" <<'PY'
import hashlib, json, pathlib, sys
root, rows, limit = pathlib.Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
result = json.loads((root / "AGENT_TEST300_COMPARISON.json").read_text(encoding="utf-8"))
if result["rows"] != rows or result["base"]["n"] != rows or result["agent_final_full_coverage"]["n"] != rows:
    raise RuntimeError("Incomplete agent test evaluation")
if result["verify_rate"] > limit + 1e-9:
    raise RuntimeError("Verification budget exceeded")
summary = {"version": "V22-ALL-plain-heterogeneous-agent-test300-complete-v1", "status": "PASS",
           "result_sha256": hashlib.sha256((root / "AGENT_TEST300_COMPARISON.json").read_bytes()).hexdigest(),
           "summary": {k: result[k] for k in ("rows", "actions", "verify_rate", "defer_rate", "coverage", "corrected", "corrupted", "net_corrections", "delta")}}
(root / "AGENT_TEST300_COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
echo "DONE: $OUT/AGENT_TEST300_COMPLETE.json"
