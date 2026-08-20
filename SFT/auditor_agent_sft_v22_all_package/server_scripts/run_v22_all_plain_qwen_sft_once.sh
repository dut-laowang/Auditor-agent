#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
GPU="${GPU:-0}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
DATA="$RUN/base_dataset"
TRACK_INDEX="$RUN/track_index"
SEALED="$RUN/modernbert_sealed_test_source"
SEALED_BUNDLE="${V22_ALL_SEALED_TEST_BUNDLE:-}"
MODEL="$RUN/models/qwen3_8b_plain_sft"
VAL="$RUN/qwen3_8b_plain_sft_validation"
TEST="$RUN/qwen3_8b_plain_sft_test"
LOGS="$RUN/logs"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO"
for path in "$DATA/train.jsonl" "$DATA/validation.jsonl" \
  "$TRACK_INDEX/train.jsonl" "$TRACK_INDEX/validation.jsonl"; do
  test -f "$path"
done
[[ "$(wc -l < "$DATA/train.jsonl" | tr -d ' ')" -eq 24204 ]]
[[ "$(wc -l < "$DATA/validation.jsonl" | tr -d ' ')" -eq 5573 ]]
mkdir -p "$LOGS"

# Plain SFT: original gold audit JSON only. No teacher expansion, ModernBERT
# predictions, structured controls, verdict head, or initialization adapter.
if [[ ! -f "$MODEL/run_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --data-dir "$DATA" --output-dir "$MODEL" \
    --max-len 12288 --context-contract v22-all-12288 --prompt-overflow error \
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
    "model": "Qwen/Qwen3-8B",
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "seed": 42,
    "max_length": 12288,
    "context_contract": "v22-all-12288",
    "prompt_overflow": "error",
    "cudnn_sdp_enabled": False,
    "train_sha256": sha(data / "train.jsonl"),
    "validation_sha256": sha(data / "validation.jsonl"),
    "test_accessed": False,
    "init_adapter": None,
}
for key, value in required.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"Plain SFT frozen-model contract mismatch: {key}")
PY

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$MODEL" --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$VAL" --max-input-len 12288 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp \
  2>&1 | tee -a "$LOGS/qwen3_8b_plain_sft_validation.log"

python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$VAL/predictions.jsonl" --track-index "$TRACK_INDEX/validation.jsonl" \
  --output-dir "$VAL/by_track"

# Open the separately transported frozen final test only after training and
# validation finish.  It is deliberately not versioned with the training data.
if [[ ! -f "$SEALED/SEALED_TEST_MANIFEST.json" ]]; then
  test -n "$SEALED_BUNDLE"; test -f "$SEALED_BUNDLE"
  mkdir -p "$SEALED"
  unzip -q "$SEALED_BUNDLE" -d "$SEALED"
fi
[[ "$(wc -l < "$SEALED/test.jsonl" | tr -d ' ')" -eq 4896 ]]

CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --adapter "$MODEL" --test-file "$SEALED/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
  --output-dir "$TEST" --max-input-len 12288 --max-new-tokens 1400 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume --disable-cudnn-sdp \
  2>&1 | tee -a "$LOGS/qwen3_8b_plain_sft_test.log"

python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$TEST/predictions.jsonl" --track-index "$SEALED/track_index.jsonl" \
  --output-dir "$TEST/by_track"

python - "$VAL" "$TEST" "$MODEL/run_manifest.json" <<'PY'
import hashlib, json, pathlib, sys
validation, test, model_manifest = map(pathlib.Path, sys.argv[1:])
val = json.loads((validation / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
tst = json.loads((test / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
if val["all"]["n"] != 5573 or tst["all"]["n"] != 4896:
    raise RuntimeError("Plain SFT evaluation is incomplete")

def compact(item):
    return {
        "n": item["n"],
        "three_class_accuracy": item["three_class_accuracy"],
        "three_class_macro_f1": item["three_class_report"]["macro avg"]["f1-score"],
        "attack_success_recall": item["three_class_report"]["attack_success"]["recall"],
        "binary_accuracy": item["binary_accuracy"],
        "localization_micro_f1": item["localization"]["component_micro_f1"],
    }

comparison = {
    "version": "V22-ALL-plain-Qwen3-8B-SFT-validation-test-v1",
    "status": "PASS",
    "method": "plain Qwen3-8B LoRA SFT; no teacher expansion or ModernBERT controls",
    "model_train_sha256": manifest["train_sha256"],
    "overall": {},
    "by_track": {},
}
for scope, val_item, test_item in [
    ("overall", val["all"], tst["all"]),
    *[(track, val["tracks"][track], tst["tracks"][track]) for track in sorted(tst["tracks"])],
]:
    left, right = compact(val_item), compact(test_item)
    entry = {
        "validation": left,
        "test": right,
        "test_minus_validation": {key: right[key] - left[key] for key in left if key != "n"},
    }
    (comparison["overall"] if scope == "overall" else comparison["by_track"]).update(
        entry if scope == "overall" else {scope: entry}
    )
(test / "V22_ALL_PLAIN_QWEN_VALIDATION_TEST_COMPARISON.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)
labels = {
    "three_class_accuracy": "3-class accuracy",
    "three_class_macro_f1": "3-class macro-F1",
    "attack_success_recall": "attack-success recall",
    "binary_accuracy": "binary accuracy",
    "localization_micro_f1": "localization micro-F1",
}
lines = [
    "# V22-ALL plain Qwen3-8B SFT: validation vs final test", "",
    "No Qwen3-32B teacher expansion or ModernBERT structured controls are used.", "",
    "| Scope | Rows (val/test) | Metric | Validation | Test | Delta |",
    "| --- | ---: | --- | ---: | ---: | ---: |",
]
for scope, entry in [("overall", comparison["overall"]), *comparison["by_track"].items()]:
    for key, label in labels.items():
        lines.append(
            f"| {scope} | {entry['validation']['n']}/{entry['test']['n']} | {label} | "
            f"{entry['validation'][key]:.6f} | {entry['test'][key]:.6f} | "
            f"{entry['test_minus_validation'][key]:+.6f} |"
        )
(test / "V22_ALL_PLAIN_QWEN_VALIDATION_TEST_COMPARISON.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
(test / "PLAIN_QWEN_TEST_COMPLETE.json").write_text(json.dumps({
    "status": "PASS",
    "validation_rows": 5573,
    "test_rows": 4896,
    "comparison_sha256": hashlib.sha256(
        (test / "V22_ALL_PLAIN_QWEN_VALIDATION_TEST_COMPARISON.json").read_bytes()
    ).hexdigest(),
}, indent=2), encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY
