#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
GPU="${GPU:-0}"
ROWS="${AGENT_DEMO_ROWS:-100}"
PLAIN_RATE="${PLAIN_AGENT_MAX_RECHECK_RATE:-0.12}"
CASCADE_RATE="${CASCADE_AGENT_MAX_RECHECK_RATE:-0.08}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
DATA="$RUN/base_dataset/validation.jsonl"
ENRICHED_DATA="$RUN/qwen_ready_audit_sft/validation.jsonl"
INDEX="$RUN/track_index/validation.jsonl"
PLAIN_PRED="$RUN/qwen3_8b_plain_sft_validation/predictions.jsonl"
CASCADE_PRED="$RUN/qwen_eval/predictions.jsonl"
CONTROLS="$RUN/modernbert_eval/predictions.jsonl"
PLAIN_MODEL="$RUN/models/qwen3_8b_plain_sft"
CASCADE_MODEL="$RUN/models/qwen3_8b_audit_grade"
OUT="$RUN/agent_demo_100"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
for path in "$DATA" "$ENRICHED_DATA" "$INDEX" "$PLAIN_PRED" "$CASCADE_PRED" "$CONTROLS" \
  "$PLAIN_MODEL/run_manifest.json" "$CASCADE_MODEL/run_manifest.json"; do test -f "$path"; done
mkdir -p "$OUT/plain" "$OUT/cascade"

prepare_and_run() {
  local variant="$1" data="$2" predictions="$3" model="$4" rate="$5" controls="${6:-}"
  local dir="$OUT/$variant"
  local control_args=() eval_control_args=()
  if [[ -n "$controls" ]]; then
    control_args+=(--controls "$controls")
    eval_control_args+=(--structured-controls "$dir/recheck_controls.jsonl")
  fi
  python "$ALL/scripts/v22_agent_demo.py" prepare --variant "$variant" --data "$data" \
    --track-index "$INDEX" --predictions "$predictions" --output-dir "$dir" \
    --rows "$ROWS" --max-recheck-rate "$rate" "${control_args[@]}"
  if [[ -s "$dir/recheck_data.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
      --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
      --adapter "$model" --test-file "$dir/recheck_data.jsonl" --dataset-role validation \
      --output-dir "$dir/recheck_eval" --max-input-len 12288 --max-new-tokens 1400 \
      --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp "${eval_control_args[@]}"
  else
    mkdir -p "$dir/recheck_eval"
    : > "$dir/recheck_eval/predictions.jsonl"
  fi
  python "$ALL/scripts/v22_agent_demo.py" merge --variant "$variant" \
    --base-predictions "$dir/base_predictions.jsonl" \
    --recheck-predictions "$dir/recheck_eval/predictions.jsonl" --output-dir "$dir"
  python "$ALL/scripts/score_predictions_by_track.py" --predictions "$dir/agent_predictions.jsonl" \
    --track-index "$dir/demo_track_index.jsonl" --output-dir "$dir/by_track"
}

prepare_and_run plain "$DATA" "$PLAIN_PRED" "$PLAIN_MODEL" "$PLAIN_RATE"
prepare_and_run cascade "$ENRICHED_DATA" "$CASCADE_PRED" "$CASCADE_MODEL" "$CASCADE_RATE" "$CONTROLS"
python - "$OUT" "$ROWS" "$PLAIN_RATE" "$CASCADE_RATE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_rows = int(sys.argv[2])
limits = {"plain": float(sys.argv[3]), "cascade": float(sys.argv[4])}
items = {}
for variant in ("plain", "cascade"):
    path = root / variant / "AGENT_DEMO_COMPARISON.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["rows"] != expected_rows or report["rows"] != int(report["base"]["n"]) or report["rows"] != int(report["agent_final"]["n"]):
        raise RuntimeError(f"Incomplete {variant} demo")
    if report["recheck_rate"] > limits[variant] + 1e-9:
        raise RuntimeError(f"Recheck-rate contract violated: {variant}")
    items[variant] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": {
        key: report[key] for key in ("rows", "recheck_rows", "recheck_rate", "corrected", "corrupted", "net_corrections", "delta")
    }}
plain_ids = [json.loads(line)["run_id"] for line in (root / "plain/demo_track_index.jsonl").open(encoding="utf-8")]
cascade_ids = [json.loads(line)["run_id"] for line in (root / "cascade/demo_track_index.jsonl").open(encoding="utf-8")]
if plain_ids != cascade_ids:
    raise RuntimeError("Plain and cascade demos did not use the exact same ordered 100 rows")
(root / "AGENT_DEMO_100_COMPLETE.json").write_text(json.dumps({
    "version": "V22-ALL-two-bounded-agent-demo-v1", "status": "PASS", "variants": items
}, indent=2), encoding="utf-8")
print(json.dumps(items, indent=2))
PY
echo "DONE: $OUT/AGENT_DEMO_100_COMPLETE.json"
