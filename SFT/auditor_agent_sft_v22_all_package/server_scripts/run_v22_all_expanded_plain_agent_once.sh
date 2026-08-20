#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_expanded_run}"
GPU="${GPU:-0}"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
SOURCE_BUNDLE="${V22_ALL_SOURCE_BUNDLE:-$ALL/source_bundle/v22_all_source_bundle.zip}"
SEALED_BUNDLE="${V22_ALL_SEALED_TEST_BUNDLE:?Set V22_ALL_SEALED_TEST_BUNDLE to the separately transferred frozen-test ZIP}"
DATA="$RUN/base_dataset"
INDEX="$RUN/track_index"
MODERN_DATA="$RUN/modernbert_data"
MODERN_MODEL="$RUN/models/modernbert_joint"
MODERN_EVAL="$RUN/modernbert_eval"
MODERN_TEST="$RUN/modernbert_sealed_test"
SEALED="$RUN/modernbert_sealed_test_source"
LOGS="$RUN/logs"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$REPO"; mkdir -p "$RUN" "$LOGS" "$RUN/models"

if [[ ! -f "$RUN/SOURCE_RESTORED.json" ]]; then
  test -f "$SOURCE_BUNDLE"
  python - "$SOURCE_BUNDLE" "$RUN" <<'PY'
import hashlib, json, pathlib, shutil, sys, tempfile, zipfile
archive, run = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(dir=run) as tmp:
    root = pathlib.Path(tmp)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    manifests = list(root.rglob("SOURCE_MANIFEST.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one SOURCE_MANIFEST.json, got {manifests}")
    source = manifests[0].parent
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("version") != "V22-ALL-unified-source-v3-expanded-2x2":
        raise RuntimeError("Wrong expanded source-bundle version")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for split in ("train", "validation"):
        expected = manifest["combined"][split]
        if digest(source / "base_dataset" / f"{split}.jsonl") != expected["sha256"]:
            raise RuntimeError(f"{split} source hash mismatch")
        if digest(source / "track_index" / f"{split}.jsonl") != expected["index_sha256"]:
            raise RuntimeError(f"{split} track-index hash mismatch")
    for name in ("base_dataset", "track_index"):
        target = run / name
        if target.exists():
            raise RuntimeError(f"Refusing existing incomplete target: {target}")
        shutil.copytree(source / name, target)
    (run / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run / "SOURCE_RESTORED.json").write_text(json.dumps({
        "status": "PASS", "source_bundle_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "train_rows": manifest["combined"]["train"]["rows"],
        "validation_rows": manifest["combined"]["validation"]["rows"],
    }, indent=2), encoding="utf-8")
PY
fi
[[ "$(wc -l < "$DATA/train.jsonl" | tr -d ' ')" -eq 24204 ]]
[[ "$(wc -l < "$DATA/validation.jsonl" | tr -d ' ')" -eq 5573 ]]

if [[ ! -f "$MODERN_DATA/MODERNBERT_CONTEXT_GATE.json" ]]; then
  python "$ALL/scripts/filter_v22_expanded_modernbert_context.py" \
    --data-dir "$DATA" --index-dir "$INDEX" --output-dir "$MODERN_DATA" \
    --splits train validation --model answerdotai/ModernBERT-base \
    --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --max-len 8192
fi
MODERN_TRAIN_SHA="$(sha256sum "$MODERN_DATA/train.jsonl" | awk '{print $1}')"
MODERN_VALIDATION_SHA="$(sha256sum "$MODERN_DATA/validation.jsonl" | awk '{print $1}')"
[[ "$(wc -l < "$MODERN_DATA/train.jsonl" | tr -d ' ')" -eq 24153 ]]
[[ "$(wc -l < "$MODERN_DATA/validation.jsonl" | tr -d ' ')" -eq 5541 ]]

if [[ ! -f "$MODERN_MODEL/TRAINING_COMPLETE.json" ]]; then
  mkdir -p "$MODERN_MODEL"
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --data-file "$MODERN_DATA/train.jsonl" --dataset-role train --output-dir "$MODERN_MODEL" \
    --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
    --max-len 8192 --attn-implementation sdpa --input-mode user --epochs "${MODERN_EPOCHS:-3}" \
    --lr "${MODERN_LR:-2e-5}" --batch "${MODERN_TRAIN_BATCH:-2}" \
    --grad-accum "${MODERN_GRAD_ACCUM:-8}" --lambda-scope 1.0 --lambda-component 1.0 --seed 42 \
    2>&1 | tee -a "$LOGS/modernbert_training.log"
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$MODERN_MODEL/checkpoint-epoch-${MODERN_EPOCHS:-3}.pt" \
  --data-file "$MODERN_DATA/validation.jsonl" --dataset-role validation \
  --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
  --output-dir "$MODERN_EVAL" --max-len 8192 --attn-implementation sdpa --input-mode user \
  --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 2>&1 | tee -a "$LOGS/modernbert_validation.log"
python "$ALL/scripts/score_predictions_by_track.py" --predictions "$MODERN_EVAL/predictions.jsonl" \
  --track-index "$MODERN_DATA/validation_track_index.jsonl" --output-dir "$MODERN_EVAL/by_track"

# Train/evaluate the primary Plain-Qwen auditor.  This opens the separately
# transported sealed test only after both model-training stages are complete.
V22_ALL_SEALED_TEST_BUNDLE="$SEALED_BUNDLE" \
  bash "$ALL/server_scripts/run_v22_all_plain_qwen_sft_once.sh"

if [[ ! -f "$MODERN_TEST/MODERNBERT_CONTEXT_GATE.json" ]]; then
  python "$ALL/scripts/filter_v22_expanded_modernbert_context.py" \
    --data-dir "$SEALED" --index-dir "$SEALED" --output-dir "$MODERN_TEST" --splits test \
    --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --max-len 8192
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$MODERN_MODEL/checkpoint-epoch-${MODERN_EPOCHS:-3}.pt" \
  --data-file "$MODERN_TEST/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
  --expected-train-sha256 "$MODERN_TRAIN_SHA" --expected-validation-sha256 "$MODERN_VALIDATION_SHA" \
  --output-dir "$MODERN_TEST" --max-len 8192 --attn-implementation sdpa --input-mode user \
  --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 2>&1 | tee -a "$LOGS/modernbert_test.log"
python "$ALL/scripts/score_predictions_by_track.py" --predictions "$MODERN_TEST/predictions.jsonl" \
  --track-index "$MODERN_TEST/test_track_index.jsonl" --output-dir "$MODERN_TEST/by_track"

AGENT_TEST_ROWS="${AGENT_TEST_ROWS:-300}" \
  bash "$ALL/server_scripts/run_v22_plain_hetero_agent_test300_once.sh"
echo "DONE: expanded Plain-Qwen + ModernBERT verifier + bounded Agent"
