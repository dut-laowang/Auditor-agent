#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
SOURCE_INPUT="${V22_ALL_SOURCE_BUNDLE:-$BASE/v22_all_source_bundle.zip}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
V20="$REPO/SFT/auditor_agent_sft_v20_appworld_marble_package"
V22="$REPO/SFT/auditor_agent_sft_v22_appworld_package"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
TRANSPORT="$RUN/source_transport"
BASE_DATA="$RUN/base_dataset"
TRACK_INDEX="$RUN/track_index"
MODERN_DATA="$RUN/modernbert_data"
V22_DATA="$RUN/v22_data"
TEACHER_DIR="$RUN/qwen32b_teacher"
EXPANDED="$RUN/expanded_audit_sft"
MODELS="$RUN/models"
MODERN_MODEL="$MODELS/modernbert"
QWEN_MODEL="$MODELS/qwen3_8b_audit_grade"
MODERN_EVAL="$RUN/modernbert_eval"
QWEN_EVAL="$RUN/qwen_eval"
LOGS="$RUN/logs"

cd "$REPO"
mkdir -p "$RUN" "$MODELS" "$LOGS" "$MODERN_EVAL" "$QWEN_EVAL"

# Fail closed unless the local 900-row stratified pre-upload gate is intact.
if [[ ! -f "$RUN/SOURCE_MANIFEST.json" ]]; then
  mkdir -p "$TRANSPORT"
  if [[ -f "$SOURCE_INPUT" ]]; then
    unzip -q "$SOURCE_INPUT" -d "$TRANSPORT"
    mapfile -t MANIFESTS < <(find "$TRANSPORT" -name SOURCE_MANIFEST.json -type f)
    [[ "${#MANIFESTS[@]}" -eq 1 ]]
    SOURCE_ROOT="$(dirname "${MANIFESTS[0]}")"
  elif [[ -d "$SOURCE_INPUT" ]]; then
    SOURCE_ROOT="$SOURCE_INPUT"
  else
    echo "Missing V22-ALL source bundle: $SOURCE_INPUT" >&2
    exit 1
  fi
  test -f "$SOURCE_ROOT/PREUPLOAD_QUALITY_GATE.json"
  test -d "$SOURCE_ROOT/base_dataset"
  test -d "$SOURCE_ROOT/track_index"
  cp -a "$SOURCE_ROOT/base_dataset" "$BASE_DATA"
  cp -a "$SOURCE_ROOT/track_index" "$TRACK_INDEX"
  cp -a "$SOURCE_ROOT/quality_samples" "$RUN/quality_samples"
  cp "$SOURCE_ROOT/PREUPLOAD_QUALITY_GATE.json" "$RUN/PREUPLOAD_QUALITY_GATE.json"
  cp "$SOURCE_ROOT/SOURCE_MANIFEST.json" "$RUN/SOURCE_MANIFEST.json"
fi

python - "$RUN/SOURCE_MANIFEST.json" "$RUN/PREUPLOAD_QUALITY_GATE.json" "$BASE_DATA" "$TRACK_INDEX" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
gate = json.load(open(sys.argv[2], encoding="utf-8"))
base, index = pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
assert manifest["version"] == "V22-ALL-unified-source-v1"
assert manifest["preupload_quality_gate"] == "PASS"
assert manifest["sealed_test_accessed"] is False
assert gate["status"] == "PASS" and gate["sampled_rows"] == 900
assert len(gate["sample_cells"]) == 18
assert gate["full_dataset_checks"]["sealed_test_accessed"] is False
for split in ("train", "validation"):
    data = base / f"{split}.jsonl"
    idx = index / f"{split}.jsonl"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["combined"][split]
    assert digest(data) == expected["sha256"]
    assert digest(idx) == expected["index_sha256"]
    assert sum(1 for line in data.open(encoding="utf-8") if line.strip()) == expected["rows"]
    assert sum(1 for line in idx.open(encoding="utf-8") if line.strip()) == expected["rows"]
PY

# One jointly trained inspector uses the unexpanded V22-ALL base data.
if [[ ! -f "$MODERN_DATA/context_filter_report.json" ]]; then
  python "$V20/scripts/filter_qwen_context_v20.py" \
    --input-dir "$BASE_DATA" --output-dir "$MODERN_DATA" \
    --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --max-len 8192 --input-mode user
fi
MODERN_TRAIN_SHA="$(sha256sum "$MODERN_DATA/train.jsonl" | awk '{print $1}')"
MODERN_VALIDATION_SHA="$(sha256sum "$MODERN_DATA/validation.jsonl" | awk '{print $1}')"
MODERN_TRAIN_ROWS="$(wc -l < "$MODERN_DATA/train.jsonl" | tr -d ' ')"
MODERN_VALIDATION_ROWS="$(wc -l < "$MODERN_DATA/validation.jsonl" | tr -d ' ')"
[[ "$MODERN_VALIDATION_ROWS" -eq 2954 ]]

if [[ ! -f "$MODERN_MODEL/TRAINING_COMPLETE.json" ]]; then
  mkdir -p "$MODERN_MODEL"
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --data-file "$MODERN_DATA/train.jsonl" --dataset-role train --output-dir "$MODERN_MODEL" \
    --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
    --max-len 8192 --attn-implementation sdpa --input-mode user --epochs 3 --lr 2e-5 \
    --batch "${MODERN_TRAIN_BATCH:-2}" --grad-accum "${MODERN_GRAD_ACCUM:-8}" \
    --lambda-scope 1.0 --lambda-component 1.0 --seed 42 \
    2>&1 | tee -a "$LOGS/modernbert_training.log"
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$MODERN_MODEL/checkpoint-epoch-3.pt" \
  --data-file "$MODERN_DATA/validation.jsonl" --dataset-role validation \
  --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
  --output-dir "$MODERN_EVAL" --max-len 8192 --attn-implementation sdpa --input-mode user \
  --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 2>&1 | tee "$LOGS/modernbert_eval.log"

python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$MODERN_EVAL/predictions.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
  --output-dir "$MODERN_EVAL/by_track"

# Train receives gold controls; validation receives only the joint inspector predictions.
if [[ ! -f "$V22_DATA/manifest.json" ]]; then
  python "$V22/scripts/build_v22_audit_dataset.py" \
    --source-data "$MODERN_DATA" --modernbert-predictions "$MODERN_EVAL/predictions.jsonl" \
    --output-dir "$V22_DATA" --expected-train-rows "$MODERN_TRAIN_ROWS" \
    --expected-validation-rows "$MODERN_VALIDATION_ROWS" \
    --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
    --dataset-version V22-ALL-explainable-audit-v1
fi

# Qwen3-32B expands training targets only and resumes against the frozen run-id prefix.
mkdir -p "$TEACHER_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python "$V22/server_scripts/qwen32b_enrich_v22_reports.py" \
  --train-file "$V22_DATA/audit_sft/train.jsonl" --output "$TEACHER_DIR/qwen32b_enrichment.jsonl" \
  --model Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --batch-size "${TEACHER_BATCH:-4}" --max-input-len 8192 --max-new-tokens 384 \
  --expected-rows "$MODERN_TRAIN_ROWS"
if [[ ! -f "$EXPANDED/EXPANSION_CONTRACT.json" ]]; then
  python "$V22/scripts/merge_qwen32b_enrichment.py" \
    --v22-data "$V22_DATA/audit_sft" --teacher-output "$TEACHER_DIR/qwen32b_enrichment.jsonl" \
    --output-dir "$EXPANDED" --expected-train-rows "$MODERN_TRAIN_ROWS" \
    --expected-validation-rows "$MODERN_VALIDATION_ROWS"
fi

INIT_ARGS=()
QWEN_LR="${QWEN_LR:-2e-4}"
if [[ -n "${QWEN_INIT_ADAPTER:-}" ]]; then
  test -f "$QWEN_INIT_ADAPTER/run_manifest.json"
  INIT_ARGS+=(--init-adapter "$QWEN_INIT_ADAPTER")
  QWEN_LR="${QWEN_LR_WITH_INIT:-1e-4}"
fi
if [[ ! -f "$QWEN_MODEL/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --data-dir "$EXPANDED" --output-dir "$QWEN_MODEL" "${INIT_ARGS[@]}" \
    --max-len 8192 --epochs "${QWEN_EPOCHS:-2}" --lr "$QWEN_LR" \
    --batch "${QWEN_TRAIN_BATCH:-2}" --grad-accum "${QWEN_GRAD_ACCUM:-8}" \
    --seed 42 --resume auto 2>&1 | tee -a "$LOGS/qwen_training.log"
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$QWEN_MODEL" --test-file "$EXPANDED/validation.jsonl" --dataset-role validation \
  --output-dir "$QWEN_EVAL" --max-input-len 8192 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume \
  --structured-controls "$MODERN_EVAL/predictions.jsonl"

python "$V22/scripts/validate_v22_results.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --metrics "$QWEN_EVAL/metrics.json" \
  --modernbert-metrics "$MODERN_EVAL/metrics.json" --output "$RUN/V22_ALL_CORE_QUALITY_GATE.json" \
  --expected-rows "$MODERN_VALIDATION_ROWS"
python "$V22/scripts/validate_v22_enriched_reports.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --validation "$EXPANDED/validation.jsonl" \
  --output "$RUN/V22_ALL_ENRICHED_REPORT_QUALITY.json" --expected-rows "$MODERN_VALIDATION_ROWS"
python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
  --output-dir "$QWEN_EVAL/by_track"

for TRACK in marble_mab autogen_mab marble_appworld; do
  TRACK_DIR="$RUN/per_track_quality/$TRACK"
  mkdir -p "$TRACK_DIR"
  python "$ALL/scripts/subset_jsonl_by_track.py" \
    --input "$EXPANDED/validation.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
    --track "$TRACK" --output "$TRACK_DIR/validation.jsonl"
  ROWS="$(wc -l < "$TRACK_DIR/validation.jsonl" | tr -d ' ')"
  python "$V22/scripts/validate_v22_results.py" \
    --predictions "$QWEN_EVAL/by_track/$TRACK/predictions.jsonl" \
    --metrics "$QWEN_EVAL/by_track/$TRACK/metrics.json" \
    --modernbert-metrics "$MODERN_EVAL/by_track/$TRACK/metrics.json" \
    --output "$TRACK_DIR/core_quality_gate.json" --expected-rows "$ROWS"
  python "$V22/scripts/validate_v22_enriched_reports.py" \
    --predictions "$QWEN_EVAL/by_track/$TRACK/predictions.jsonl" \
    --validation "$TRACK_DIR/validation.jsonl" \
    --output "$TRACK_DIR/enriched_report_quality.json" --expected-rows "$ROWS"
done

python - "$RUN" "$MODERN_TRAIN_ROWS" "$MODERN_VALIDATION_ROWS" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {
    "version": "V22-ALL",
    "status": "PASS",
    "modernbert_train_rows_after_joint_context_gate": int(sys.argv[2]),
    "validation_rows": int(sys.argv[3]),
    "preupload_sample_rows": 900,
    "sealed_test_accessed": False,
    "artifacts": {},
}
for relative in (
    "PREUPLOAD_QUALITY_GATE.json", "SOURCE_MANIFEST.json",
    "V22_ALL_CORE_QUALITY_GATE.json", "V22_ALL_ENRICHED_REPORT_QUALITY.json",
    "modernbert_eval/by_track/metrics_by_track.json", "qwen_eval/by_track/metrics_by_track.json",
):
    path = root / relative
    assert path.is_file()
    summary["artifacts"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "V22_ALL_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

touch "$RUN/PIPELINE_COMPLETE"
ARCHIVE="$RUN.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN")" "$(basename "$RUN")"
echo "DONE: $ARCHIVE"
