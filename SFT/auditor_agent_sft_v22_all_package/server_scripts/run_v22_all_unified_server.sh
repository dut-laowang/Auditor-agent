#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
SOURCE_INPUT="${V22_ALL_SOURCE_BUNDLE:-$REPO/SFT/auditor_agent_sft_v22_all_package/source_bundle/v22_all_source_bundle.zip}"
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
PRIOR_APPWORLD_TEACHER="${V22_APPWORLD_TEACHER:-$ALL/reuse/appworld_qwen32b_enrichment_v4.jsonl}"
PRIOR_APPWORLD_CONTRACT="${V22_APPWORLD_TEACHER_CONTRACT:-$ALL/reuse/appworld_qwen32b_enrichment_v4.contract.json}"
EXPANDED="$RUN/expanded_audit_sft"
QWEN_READY="$RUN/qwen_ready_audit_sft"
MODELS="$RUN/models"
MODERN_MODEL="$MODELS/modernbert"
QWEN_MODEL="$MODELS/qwen3_8b_audit_grade"
MODERN_EVAL="$RUN/modernbert_eval"
QWEN_EVAL="$RUN/qwen_eval"
LOGS="$RUN/logs"

cd "$REPO"

require_safe_run_path() {
  case "$RUN" in
    "$BASE"/v22_all_run|"$BASE"/v22_all_run_*) ;;
    *) echo "Refusing automatic cleanup outside an exact V22-ALL run path: $RUN" >&2; exit 1 ;;
  esac
}

# Automatically quarantine only downstream artifacts produced by the obsolete
# all-row teacher stage. Source restoration, context filtering, and the joint
# ModernBERT checkpoint/evaluation remain reusable. A corrected partial run has
# APPWORLD_REUSE_CONTRACT.json and is never treated as stale here.
STALE_DOWNSTREAM=0
if [[ -f "$TEACHER_DIR/qwen32b_enrichment.jsonl" && ! -f "$TEACHER_DIR/APPWORLD_REUSE_CONTRACT.json" ]]; then
  STALE_DOWNSTREAM=1
elif [[ -f "$EXPANDED/EXPANSION_CONTRACT.json" ]]; then
  if [[ ! -f "$EXPANDED/V22_ALL_REUSE_APPLIED.json" ]] || ! grep -q 'V22-enriched-audit-v2' "$EXPANDED/EXPANSION_CONTRACT.json"; then
    STALE_DOWNSTREAM=1
  fi
fi
if [[ -d "$RUN" && "$STALE_DOWNSTREAM" -eq 1 ]]; then
  require_safe_run_path
  STALE="$BASE/v22_all_stale_pre_appworld_reuse_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$STALE"
  for TARGET in \
    "$TEACHER_DIR" "$EXPANDED" "$QWEN_READY" "$QWEN_MODEL" "$QWEN_EVAL" \
    "$RUN/per_track_quality" "$RUN/V22_ALL_CORE_QUALITY_GATE.json" \
    "$RUN/V22_ALL_ENRICHED_REPORT_QUALITY.json" "$RUN/V22_ALL_RUN_SUMMARY.json" \
    "$RUN/PIPELINE_COMPLETE" "$RUN.tar.gz"; do
    if [[ -e "$TARGET" ]]; then mv -- "$TARGET" "$STALE/"; fi
  done
  echo "Quarantined obsolete all-row teacher/downstream artifacts at: $STALE"
fi
mkdir -p "$RUN" "$MODELS" "$LOGS" "$MODERN_EVAL" "$QWEN_EVAL"

if [[ -f "$RUN/SOURCE_MANIFEST.json" ]] && ! grep -q 'V22-ALL-unified-source-v2' "$RUN/SOURCE_MANIFEST.json"; then
  require_safe_run_path
  STALE_SOURCE="$BASE/v22_all_stale_source_v1_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$STALE_SOURCE"
  for TARGET in "$BASE_DATA" "$TRACK_INDEX" "$RUN/quality_samples" "$TRANSPORT" \
    "$RUN/PREUPLOAD_QUALITY_GATE.json" "$RUN/SOURCE_MANIFEST.json"; do
    if [[ -e "$TARGET" ]]; then mv -- "$TARGET" "$STALE_SOURCE/"; fi
  done
  echo "Moved obsolete V22-ALL source gate outside the active run: $STALE_SOURCE"
fi

# Fail closed unless the local 900-row stratified pre-upload gate is intact.
if [[ ! -f "$RUN/SOURCE_MANIFEST.json" ]]; then
  RESTORE="$RUN/.source_restore_$$"
  shopt -s nullglob
  RESTORE_LEFTOVERS=("$RUN"/.source_restore_*)
  shopt -u nullglob
  if [[ -e "$BASE_DATA" || -e "$TRACK_INDEX" || -e "$RUN/quality_samples" || -e "$TRANSPORT" || "${#RESTORE_LEFTOVERS[@]}" -gt 0 ]]; then
    require_safe_run_path
    PARTIAL="$BASE/v22_all_stale_partial_source_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$PARTIAL"
    for TARGET in "$BASE_DATA" "$TRACK_INDEX" "$RUN/quality_samples" "$TRANSPORT" \
      "$RUN/PREUPLOAD_QUALITY_GATE.json" "${RESTORE_LEFTOVERS[@]}"; do
      if [[ -e "$TARGET" ]]; then mv -- "$TARGET" "$PARTIAL/"; fi
    done
    echo "Moved incomplete source restoration outside the active run: $PARTIAL"
  fi
  mkdir -p "$RESTORE/transport" "$RESTORE/staged"
  if [[ -f "$SOURCE_INPUT" ]]; then
    unzip -q -o "$SOURCE_INPUT" -d "$RESTORE/transport"
    mapfile -t MANIFESTS < <(find "$RESTORE/transport" -name SOURCE_MANIFEST.json -type f)
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
  test -d "$SOURCE_ROOT/quality_samples"
  cp -a "$SOURCE_ROOT/base_dataset" "$RESTORE/staged/base_dataset"
  cp -a "$SOURCE_ROOT/track_index" "$RESTORE/staged/track_index"
  cp -a "$SOURCE_ROOT/quality_samples" "$RESTORE/staged/quality_samples"
  cp "$SOURCE_ROOT/PREUPLOAD_QUALITY_GATE.json" "$RESTORE/staged/PREUPLOAD_QUALITY_GATE.json"
  cp "$SOURCE_ROOT/SOURCE_MANIFEST.json" "$RESTORE/staged/SOURCE_MANIFEST.json"
  mv -- "$RESTORE/transport" "$TRANSPORT"
  mv -- "$RESTORE/staged/base_dataset" "$BASE_DATA"
  mv -- "$RESTORE/staged/track_index" "$TRACK_INDEX"
  mv -- "$RESTORE/staged/quality_samples" "$RUN/quality_samples"
  mv -- "$RESTORE/staged/PREUPLOAD_QUALITY_GATE.json" "$RUN/PREUPLOAD_QUALITY_GATE.json"
  mv -- "$RESTORE/staged/SOURCE_MANIFEST.json" "$RUN/SOURCE_MANIFEST.json"
  rmdir "$RESTORE/staged" "$RESTORE"
fi

python - "$RUN/SOURCE_MANIFEST.json" "$RUN/PREUPLOAD_QUALITY_GATE.json" "$BASE_DATA" "$TRACK_INDEX" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
gate = json.load(open(sys.argv[2], encoding="utf-8"))
base, index = pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(manifest["version"] == "V22-ALL-unified-source-v2", "Wrong source manifest version")
require(manifest["preupload_quality_gate"] == "PASS", "Source quality gate is not PASS")
require(manifest["sealed_test_accessed"] is False, "Sealed test access flag is not false")
require(gate["status"] == "PASS" and gate["sampled_rows"] == 900, "900-row gate is not PASS")
require(len(gate["sample_cells"]) == 18, "Expected 18 sample cells")
require(all(cell.get("semantic_contract_checks") == "PASS" for cell in gate["sample_cells"].values()), "Semantic sample gate is incomplete")
require(gate["full_dataset_checks"]["sealed_test_accessed"] is False, "Sealed test access flag is not false")
for split in ("train", "validation"):
    data = base / f"{split}.jsonl"
    idx = index / f"{split}.jsonl"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["combined"][split]
    require(digest(data) == expected["sha256"], f"{split} data hash mismatch")
    require(digest(idx) == expected["index_sha256"], f"{split} index hash mismatch")
    require(sum(1 for line in data.open(encoding="utf-8") if line.strip()) == expected["rows"], f"{split} data row mismatch")
    require(sum(1 for line in idx.open(encoding="utf-8") if line.strip()) == expected["rows"], f"{split} index row mismatch")
PY
SOURCE_TRAIN_SHA="$(sha256sum "$BASE_DATA/train.jsonl" | awk '{print $1}')"
SOURCE_VALIDATION_SHA="$(sha256sum "$BASE_DATA/validation.jsonl" | awk '{print $1}')"
SOURCE_TRAIN_ROWS="$(wc -l < "$BASE_DATA/train.jsonl" | tr -d ' ')"
SOURCE_VALIDATION_ROWS="$(wc -l < "$BASE_DATA/validation.jsonl" | tr -d ' ')"
[[ "$SOURCE_TRAIN_ROWS" -eq 10438 ]]
[[ "$SOURCE_VALIDATION_ROWS" -eq 2954 ]]

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
python - "$MODERN_DATA/context_filter_report.json" "$BASE_DATA" "$MODERN_DATA" <<'PY'
import hashlib, json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
source, filtered = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for split in ("train", "validation"):
    item = report["splits"][split]
    if item["source_sha256"] != sha(source / f"{split}.jsonl"):
        raise RuntimeError(f"ModernBERT context-gate source mismatch: {split}")
    if item["filtered_sha256"] != sha(filtered / f"{split}.jsonl"):
        raise RuntimeError(f"ModernBERT context-gate output mismatch: {split}")
if report["splits"]["validation"]["dropped_rows"] != 0:
    raise RuntimeError("ModernBERT context gate dropped validation rows")
PY

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

# Validation inference is deterministic and expensive. Reuse it only after
# verifying the exact validation data, checkpoint, role, and complete row count.
MODERN_EVAL_VALID=0
if [[ -f "$MODERN_EVAL/metrics.json" && -f "$MODERN_EVAL/predictions.jsonl" ]]; then
  if python - "$MODERN_EVAL/metrics.json" "$MODERN_EVAL/predictions.jsonl" \
      "$MODERN_DATA/validation.jsonl" "$MODERN_MODEL/checkpoint-epoch-3.pt" \
      "$MODERN_VALIDATION_ROWS" <<'PY'
import hashlib, json, pathlib, sys
metrics_path, predictions, validation, checkpoint = map(pathlib.Path, sys.argv[1:5])
expected_rows = int(sys.argv[5])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if metrics.get("dataset_role") != "validation":
    raise SystemExit(1)
if metrics.get("data_sha256") != sha(validation):
    raise SystemExit(1)
if metrics.get("checkpoint_sha256") != sha(checkpoint):
    raise SystemExit(1)
if sum(1 for line in predictions.open(encoding="utf-8") if line.strip()) != expected_rows:
    raise SystemExit(1)
PY
  then
    MODERN_EVAL_VALID=1
  fi
fi
if [[ "$MODERN_EVAL_VALID" -ne 1 ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --checkpoint "$MODERN_MODEL/checkpoint-epoch-3.pt" \
    --data-file "$MODERN_DATA/validation.jsonl" --dataset-role validation \
    --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
    --output-dir "$MODERN_EVAL" --max-len 8192 --attn-implementation sdpa --input-mode user \
    --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 2>&1 | tee "$LOGS/modernbert_eval.log"
else
  echo "Reusing verified ModernBERT validation predictions: $MODERN_EVAL/predictions.jsonl"
fi

python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$MODERN_EVAL/predictions.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
  --output-dir "$MODERN_EVAL/by_track"

# ModernBERT alone excludes its 18 overlength training documents. The complete
# source split remains intact for V22/Qwen because training controls are gold;
# validation receives only the joint inspector predictions.
V22_DATA_VALID=0
if [[ -f "$V22_DATA/manifest.json" ]]; then
  if python - "$V22_DATA/manifest.json" "$SOURCE_TRAIN_SHA" "$SOURCE_VALIDATION_SHA" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("train_rows") != 10438 or manifest.get("validation_rows") != 2954:
    raise SystemExit(1)
if manifest.get("train_source_sha256") != sys.argv[2] or manifest.get("validation_source_sha256") != sys.argv[3]:
    raise SystemExit(1)
PY
  then
    V22_DATA_VALID=1
  fi
fi
if [[ -f "$V22_DATA/manifest.json" && "$V22_DATA_VALID" -ne 1 ]]; then
  require_safe_run_path
  STALE_V22="$BASE/v22_all_stale_partial_v22_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$STALE_V22"
  for TARGET in "$V22_DATA" "$TEACHER_DIR" "$EXPANDED" "$QWEN_READY" "$QWEN_MODEL" "$QWEN_EVAL" \
    "$RUN/per_track_quality" "$RUN/V22_ALL_CORE_QUALITY_GATE.json" \
    "$RUN/V22_ALL_ENRICHED_REPORT_QUALITY.json" "$RUN/V22_ALL_RUN_SUMMARY.json" \
    "$RUN/PIPELINE_COMPLETE" "$RUN.tar.gz"; do
    if [[ -e "$TARGET" ]]; then mv -- "$TARGET" "$STALE_V22/"; fi
  done
  echo "Moved obsolete reduced V22/Qwen artifacts outside the active run: $STALE_V22"
fi
if [[ ! -f "$V22_DATA/manifest.json" ]]; then
  python "$V22/scripts/build_v22_audit_dataset.py" \
    --source-data "$BASE_DATA" --modernbert-predictions "$MODERN_EVAL/predictions.jsonl" \
    --output-dir "$V22_DATA" --expected-train-rows "$SOURCE_TRAIN_ROWS" \
    --expected-validation-rows "$SOURCE_VALIDATION_ROWS" \
    --expected-train-sha256 "$SOURCE_TRAIN_SHA" --expected-validation-sha256 "$SOURCE_VALIDATION_SHA" \
    --dataset-version V22-ALL-explainable-audit-v2
fi

# Reuse the already completed 3,122-row AppWorld teacher expansion. The strict
# source hash and exact ID/order checks fail closed before any new inference.
mkdir -p "$TEACHER_DIR"
test -f "$PRIOR_APPWORLD_TEACHER"
test -f "$PRIOR_APPWORLD_CONTRACT"
python "$ALL/scripts/reuse_v22_appworld_teacher.py" prepare \
  --v22-train "$V22_DATA/audit_sft/train.jsonl" \
  --track-index "$TRACK_INDEX/train.jsonl" \
  --prior-teacher "$PRIOR_APPWORLD_TEACHER" \
  --prior-contract "$PRIOR_APPWORLD_CONTRACT" \
  --appworld-train "$TEACHER_DIR/appworld_reused_train.jsonl" \
  --new-train "$TEACHER_DIR/new_two_track_train.jsonl" \
  --output-contract "$TEACHER_DIR/APPWORLD_REUSE_CONTRACT.json"
NEW_TEACHER_ROWS="$(wc -l < "$TEACHER_DIR/new_two_track_train.jsonl" | tr -d ' ')"

# Qwen3-32B runs only on MARBLE x MAB and AutoGen x MAB rows.
CUDA_VISIBLE_DEVICES="$GPU" python "$V22/server_scripts/qwen32b_enrich_v22_reports.py" \
  --train-file "$TEACHER_DIR/new_two_track_train.jsonl" --output "$TEACHER_DIR/new_two_track_enrichment.jsonl" \
  --model Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --batch-size "${TEACHER_BATCH:-2}" --max-input-len 12288 --max-new-tokens 384 \
  --expected-rows "$NEW_TEACHER_ROWS"
python "$ALL/scripts/reuse_v22_appworld_teacher.py" merge \
  --v22-train "$V22_DATA/audit_sft/train.jsonl" \
  --prior-teacher "$PRIOR_APPWORLD_TEACHER" \
  --new-teacher "$TEACHER_DIR/new_two_track_enrichment.jsonl" \
  --output "$TEACHER_DIR/qwen32b_enrichment.jsonl" \
  --output-contract "$TEACHER_DIR/V22_ALL_TEACHER_MERGE_CONTRACT.json"
if [[ -f "$EXPANDED/EXPANSION_CONTRACT.json" && ! -f "$EXPANDED/V22_ALL_REUSE_APPLIED.json" ]]; then
  echo "Refusing stale expanded data that predates AppWorld reuse; use a fresh V22_ALL_RUN." >&2
  exit 1
fi
if [[ ! -f "$EXPANDED/EXPANSION_CONTRACT.json" ]]; then
  python "$V22/scripts/merge_qwen32b_enrichment.py" \
    --v22-data "$V22_DATA/audit_sft" --teacher-output "$TEACHER_DIR/qwen32b_enrichment.jsonl" \
    --output-dir "$EXPANDED" --expected-train-rows "$SOURCE_TRAIN_ROWS" \
    --expected-validation-rows "$SOURCE_VALIDATION_ROWS"
  cp "$TEACHER_DIR/V22_ALL_TEACHER_MERGE_CONTRACT.json" "$EXPANDED/V22_ALL_REUSE_APPLIED.json"
fi

# A resumed expansion is reusable only when its current inputs and outputs
# exactly match the recorded contract.
python - "$EXPANDED" "$V22_DATA/audit_sft" "$TEACHER_DIR/qwen32b_enrichment.jsonl" <<'PY'
import hashlib, json, pathlib, sys
expanded, source, teacher = map(pathlib.Path, sys.argv[1:])
contract = json.loads((expanded / "EXPANSION_CONTRACT.json").read_text(encoding="utf-8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
expected = {
    "version": "V22-enriched-audit-v2",
    "source_train_sha256": sha(source / "train.jsonl"),
    "source_validation_sha256": sha(source / "validation.jsonl"),
    "teacher_output_sha256": sha(teacher),
    "expanded_train_sha256": sha(expanded / "train.jsonl"),
    "expanded_validation_sha256": sha(expanded / "validation.jsonl"),
}
for key, value in expected.items():
    if contract.get(key) != value:
        raise RuntimeError(f"Expansion resume contract mismatch: {key}")
PY

# The exact complete enriched chat, not the pre-expansion source, determines
# Qwen SFT eligibility. Qwen3-8B's pinned config supports 40,960 positions;
# V22-ALL uses a conservative 12,288-token budget and preserves every row.
# Any over-budget train or validation chat stops with an ID-level report.
if [[ ! -f "$QWEN_READY/FINAL_QWEN_CONTEXT_GATE.json" ]]; then
  python "$ALL/scripts/filter_final_qwen_context.py" \
    --input-dir "$EXPANDED" --output-dir "$QWEN_READY" \
    --track-index "$TRACK_INDEX/train.jsonl" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --max-len 12288 --expected-validation-rows "$SOURCE_VALIDATION_ROWS"
fi
QWEN_TRAIN_ROWS="$(wc -l < "$QWEN_READY/train.jsonl" | tr -d ' ')"
[[ "$QWEN_TRAIN_ROWS" -eq "$SOURCE_TRAIN_ROWS" ]]

python - "$QWEN_READY" "$EXPANDED" "$MODERN_VALIDATION_ROWS" <<'PY'
import hashlib, json, pathlib, sys
ready, expanded, expected_validation = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
report = json.loads((ready / "FINAL_QWEN_CONTEXT_GATE.json").read_text(encoding="utf-8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if report.get("status") != "PASS":
    raise RuntimeError("Final Qwen context gate is not PASS")
if report["splits"]["train"]["source_sha256"] != sha(expanded / "train.jsonl"):
    raise RuntimeError("Final Qwen train source hash mismatch")
if report["splits"]["validation"]["source_sha256"] != sha(expanded / "validation.jsonl"):
    raise RuntimeError("Final Qwen validation source hash mismatch")
if report["splits"]["train"]["filtered_sha256"] != sha(ready / "train.jsonl"):
    raise RuntimeError("Final Qwen train output hash mismatch")
if report["splits"]["validation"]["filtered_sha256"] != sha(ready / "validation.jsonl"):
    raise RuntimeError("Final Qwen validation output hash mismatch")
if report["splits"]["validation"]["kept_rows"] != expected_validation:
    raise RuntimeError("Final Qwen validation rows changed")
PY

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
    --data-dir "$QWEN_READY" --output-dir "$QWEN_MODEL" "${INIT_ARGS[@]}" \
    --max-len 12288 --epochs "${QWEN_EPOCHS:-2}" --lr "$QWEN_LR" \
    --batch "${QWEN_TRAIN_BATCH:-1}" --grad-accum "${QWEN_GRAD_ACCUM:-16}" \
    --seed 42 --resume auto 2>&1 | tee -a "$LOGS/qwen_training.log"
fi
python - "$QWEN_MODEL/run_manifest.json" "$QWEN_READY" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
data = pathlib.Path(sys.argv[2])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if manifest.get("train_sha256") != sha(data / "train.jsonl"):
    raise RuntimeError("Qwen model/train data resume hash mismatch")
if manifest.get("validation_sha256") != sha(data / "validation.jsonl"):
    raise RuntimeError("Qwen model/validation data resume hash mismatch")
PY
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$QWEN_MODEL" --test-file "$QWEN_READY/validation.jsonl" --dataset-role validation \
  --output-dir "$QWEN_EVAL" --max-input-len 8192 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume \
  --structured-controls "$MODERN_EVAL/predictions.jsonl"

python "$V22/scripts/validate_v22_results.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --metrics "$QWEN_EVAL/metrics.json" \
  --modernbert-metrics "$MODERN_EVAL/metrics.json" --output "$RUN/V22_ALL_CORE_QUALITY_GATE.json" \
  --expected-rows "$MODERN_VALIDATION_ROWS"
python "$V22/scripts/validate_v22_enriched_reports.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --validation "$QWEN_READY/validation.jsonl" \
  --output "$RUN/V22_ALL_ENRICHED_REPORT_QUALITY.json" --expected-rows "$MODERN_VALIDATION_ROWS"
python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$QWEN_EVAL/predictions.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
  --output-dir "$QWEN_EVAL/by_track"

for TRACK in marble_mab autogen_mab marble_appworld; do
  TRACK_DIR="$RUN/per_track_quality/$TRACK"
  mkdir -p "$TRACK_DIR"
  python "$ALL/scripts/subset_jsonl_by_track.py" \
    --input "$QWEN_READY/validation.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
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

python - "$RUN" "$MODERN_TRAIN_ROWS" "$QWEN_TRAIN_ROWS" "$MODERN_VALIDATION_ROWS" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
enriched_paths = [root / "V22_ALL_ENRICHED_REPORT_QUALITY.json"] + [
    root / "per_track_quality" / track / "enriched_report_quality.json"
    for track in ("marble_mab", "autogen_mab", "marble_appworld")
]
for path in enriched_paths:
    require(path.is_file(), f"Missing enriched quality gate: {path}")
    require(json.loads(path.read_text(encoding="utf-8")).get("quality_gate") == "PASS", f"Failed enriched quality gate: {path}")
for track in ("marble_mab", "autogen_mab", "marble_appworld"):
    require((root / "per_track_quality" / track / "core_quality_gate.json").is_file(), f"Missing core gate: {track}")
summary = {
    "version": "V22-ALL",
    "status": "PASS",
    "modernbert_train_rows_after_joint_context_gate": int(sys.argv[2]),
    "qwen_train_rows_after_final_enriched_context_gate": int(sys.argv[3]),
    "validation_rows": int(sys.argv[4]),
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
    require(path.is_file(), f"Missing required final artifact: {relative}")
    summary["artifacts"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "V22_ALL_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

touch "$RUN/PIPELINE_COMPLETE"
ARCHIVE="$RUN.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUN")" "$(basename "$RUN")"
echo "DONE: $ARCHIVE"
