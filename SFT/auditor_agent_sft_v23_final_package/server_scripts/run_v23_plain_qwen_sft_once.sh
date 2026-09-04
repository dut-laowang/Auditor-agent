#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V23_RUN:-$BASE/v23_final_run}"
DATA="${V23_DATA_DIR:?Set V23_DATA_DIR to the uploaded v23_final_aligned_combined directory}"
GPU="${GPU:-0}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
V22="$REPO/SFT/auditor_agent_sft_v22_all_package"
MODEL="$RUN/models/qwen3_8b_plain_sft"
VAL="$RUN/qwen3_8b_plain_sft_validation"
TEST="$RUN/qwen3_8b_plain_sft_test"
LOGS="$RUN/logs"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO"
mkdir -p "$RUN/models" "$LOGS"

# Fail closed before allocating a GPU: exact counts and hashes must match the
# uploaded final manifest. Test content is not opened until training finishes.
python - "$DATA" <<'PY'
import hashlib, json, pathlib, sys
data = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads((data / "COMBINED_MANIFEST.json").read_text(encoding="utf-8"))
if manifest.get("version") != "V23-ALL-expanded-2x2-combined-final-v1":
    raise RuntimeError("Wrong V23 combined manifest version")
expected = {"train": 30619, "validation": 7018, "test": 6207}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for split in ("train", "validation"):
    entry = manifest["splits"][split]
    rows = sum(1 for line in (data / f"{split}.jsonl").open(encoding="utf-8") if line.strip())
    if rows != expected[split] or entry.get("rows") != expected[split]:
        raise RuntimeError(f"{split} row-count mismatch")
    if sha(data / f"{split}.jsonl") != entry.get("sha256"):
        raise RuntimeError(f"{split} hash mismatch")
    if sha(data / f"{split}_track_index.jsonl") != entry.get("index_sha256"):
        raise RuntimeError(f"{split} track-index hash mismatch")
PY

if [[ ! -f "$MODEL/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --data-dir "$DATA" --output-dir "$MODEL" \
    --max-len 12288 --context-contract v23-all-12288 --prompt-overflow error \
    --epochs "${QWEN_EPOCHS:-2}" --lr "${QWEN_LR:-2e-4}" \
    --batch "${QWEN_TRAIN_BATCH:-1}" --grad-accum "${QWEN_GRAD_ACCUM:-16}" \
    --seed 42 --resume auto --disable-cudnn-sdp \
    2>&1 | tee -a "$LOGS/qwen3_8b_plain_sft_training.log"
fi

python - "$MODEL/run_manifest.json" "$DATA" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
data = pathlib.Path(sys.argv[2])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
required = {
    "version": "V23-ALL-audit-grade-sft-v1",
    "model": "Qwen/Qwen3-8B",
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "seed": 42, "max_length": 12288,
    "context_contract": "v23-all-12288", "prompt_overflow": "error",
    "cudnn_sdp_enabled": False,
    "train_sha256": sha(data / "train.jsonl"),
    "validation_sha256": sha(data / "validation.jsonl"),
    "test_accessed": False, "init_adapter": None,
}
for key, value in required.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"V23 SFT resume contract mismatch: {key}")
PY

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$MODEL" --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$VAL" --max-input-len 12288 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp \
  2>&1 | tee -a "$LOGS/qwen3_8b_plain_sft_validation.log"
python "$V22/scripts/score_predictions_by_track.py" --predictions "$VAL/predictions.jsonl" \
  --track-index "$DATA/validation_track_index.jsonl" --output-dir "$VAL/by_track"

# Final test is opened only after training and validation have completed.
python - "$DATA" <<'PY'
import hashlib, json, pathlib, sys
data = pathlib.Path(sys.argv[1]); manifest = json.loads((data / "COMBINED_MANIFEST.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
entry = manifest["splits"]["test"]
rows = sum(1 for line in (data / "test.jsonl").open(encoding="utf-8") if line.strip())
if rows != 6207 or entry.get("rows") != 6207 or sha(data / "test.jsonl") != entry.get("sha256"):
    raise RuntimeError("Final test count/hash mismatch")
if sha(data / "test_track_index.jsonl") != entry.get("index_sha256"):
    raise RuntimeError("Final test track-index hash mismatch")
PY
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$MODEL" --test-file "$DATA/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
  --output-dir "$TEST" --max-input-len 12288 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp \
  2>&1 | tee -a "$LOGS/qwen3_8b_plain_sft_test.log"
python "$V22/scripts/score_predictions_by_track.py" --predictions "$TEST/predictions.jsonl" \
  --track-index "$DATA/test_track_index.jsonl" --output-dir "$TEST/by_track"

python - "$VAL" "$TEST" "$RUN" <<'PY'
import hashlib, json, pathlib, sys
validation, test, run = map(pathlib.Path, sys.argv[1:])
val = json.loads((validation / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
tst = json.loads((test / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
if val["all"]["n"] != 7018 or tst["all"]["n"] != 6207:
    raise RuntimeError("V23 evaluation is incomplete")
summary = {"version": "V23-ALL-plain-Qwen3-8B-SFT-v1", "status": "PASS",
           "validation_rows": 7018, "test_rows": 6207,
           "validation_metrics_sha256": hashlib.sha256((validation / "by_track/metrics_by_track.json").read_bytes()).hexdigest(),
           "test_metrics_sha256": hashlib.sha256((test / "by_track/metrics_by_track.json").read_bytes()).hexdigest()}
(run / "V23_SFT_COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
echo "DONE: $RUN/V23_SFT_COMPLETE.json"
