#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_ALL_RUN:-$BASE/v22_all_run}"
GPU="${GPU:-0}"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
ALL="$REPO/SFT/auditor_agent_sft_v22_all_package"
MODEL="$RUN/models/modernbert"
CHECKPOINT="$MODEL/checkpoint-epoch-3.pt"
REFERENCE="$RUN/base_dataset"
SEALED="$RUN/modernbert_sealed_test_source"
OUTPUT="$RUN/modernbert_sealed_test"
VALIDATION="$RUN/modernbert_eval"
LOGS="$RUN/logs"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO"
test -f "$CHECKPOINT"
test -f "$MODEL/TRAINING_COMPLETE.json"
test -f "$MODEL/TRAIN_CONTRACT.json"
test -f "$MODEL/component_threshold.json"
test -f "$REFERENCE/train.jsonl"
test -f "$REFERENCE/validation.jsonl"
test -f "$VALIDATION/metrics.json"
test -f "$VALIDATION/by_track/metrics_by_track.json"
mkdir -p "$LOGS"

python "$ALL/scripts/prepare_v22_all_sealed_test.py" \
  --marble-mab-zip "$REPO/SFT/auditor_agent_sft_v20_marble_package/dataset_bundle/dataset_jsonl.zip" \
  --autogen-mab-zip "$REPO/SFT/auditor_agent_sft_v20_autogen_package/dataset_bundle/dataset_jsonl.zip" \
  --marble-appworld-zip "$REPO/SFT/auditor_agent_sft_v20_appworld_marble_package/dataset_bundle/dataset_jsonl.zip" \
  --reference-data "$REFERENCE" --output-dir "$SEALED"

TRAIN_SHA="$(sha256sum "$RUN/modernbert_data/train.jsonl" | awk '{print $1}')"
VALIDATION_SHA="$(sha256sum "$RUN/modernbert_data/validation.jsonl" | awk '{print $1}')"
TEST_SHA="$(sha256sum "$SEALED/test.jsonl" | awk '{print $1}')"

if [[ ! -f "$OUTPUT/FINAL_TEST_COMPLETE.json" ]]; then
  mkdir -p "$OUTPUT"
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --checkpoint "$CHECKPOINT" \
    --data-file "$SEALED/test.jsonl" --dataset-role test --sealed-test-ack FINAL_ONCE \
    --expected-train-sha256 "$TRAIN_SHA" --expected-validation-sha256 "$VALIDATION_SHA" \
    --expected-test-sha256 "$TEST_SHA" \
    --output-dir "$OUTPUT" --max-len 8192 --attn-implementation sdpa --input-mode user \
    --batch "${MODERN_TEST_BATCH:-4}" --seed 42 2>&1 | tee "$LOGS/modernbert_sealed_test.log"
else
  echo "Reusing completed final ModernBERT test: $OUTPUT/FINAL_TEST_COMPLETE.json"
fi

python "$ALL/scripts/score_predictions_by_track.py" \
  --predictions "$OUTPUT/predictions.jsonl" --track-index "$SEALED/track_index.jsonl" \
  --output-dir "$OUTPUT/by_track"

python - "$OUTPUT" "$SEALED/SEALED_TEST_MANIFEST.json" "$VALIDATION" <<'PY'
import hashlib, json, pathlib, sys
output, manifest_path, validation = map(pathlib.Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
by_track = json.loads((output / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
validation_by_track = json.loads((validation / "by_track/metrics_by_track.json").read_text(encoding="utf-8"))
if metrics.get("dataset_role") != "test" or metrics.get("n") != 2539:
    raise RuntimeError("Final test metrics have the wrong role or row count")
if metrics.get("data_sha256") != manifest["test_sha256"]:
    raise RuntimeError("Final test metrics/data hash mismatch")
summary = {
    "version": "V22-ALL-ModernBERT-final-test-v1",
    "status": "PASS",
    "rows": 2539,
    "tracks": manifest["tracks"],
    "checkpoint_sha256": metrics["checkpoint_sha256"],
    "test_sha256": metrics["data_sha256"],
    "metrics_sha256": sha(output / "metrics.json"),
    "predictions_sha256": sha(output / "predictions.jsonl"),
    "overall": by_track["all"],
    "by_track": by_track["tracks"],
    "sealed_test_accessed": True,
}
(output / "V22_ALL_BERT_TEST_SUMMARY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

def compact(item):
    return {
        "n": item["n"],
        "three_class_accuracy": item["three_class_accuracy"],
        "three_class_macro_f1": item["three_class_report"]["macro avg"]["f1-score"],
        "attack_success_recall": item["three_class_report"]["attack_success"]["recall"],
        "binary_accuracy": item["binary_accuracy"],
        "localization_micro_f1": item["localization"]["component_micro_f1"],
    }

comparison = {"version": "V22-ALL-ModernBERT-validation-test-comparison-v1", "overall": {}, "by_track": {}}
for scope, validation_item, test_item in [
    ("overall", validation_by_track["all"], by_track["all"]),
    *[(track, validation_by_track["tracks"][track], by_track["tracks"][track]) for track in sorted(by_track["tracks"])],
]:
    validation_values, test_values = compact(validation_item), compact(test_item)
    entry = {
        "validation": validation_values,
        "test": test_values,
        "test_minus_validation": {
            key: test_values[key] - validation_values[key]
            for key in validation_values if key != "n"
        },
    }
    if scope == "overall":
        comparison["overall"] = entry
    else:
        comparison["by_track"][scope] = entry
(output / "V22_ALL_BERT_VALIDATION_TEST_COMPARISON.json").write_text(
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
    "# V22-ALL ModernBERT: validation vs final test",
    "",
    "Positive delta means test is higher than validation.",
    "",
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
(output / "V22_ALL_BERT_VALIDATION_TEST_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({
    "status": "PASS",
    "rows": summary["rows"],
    "three_class_accuracy": summary["overall"]["three_class_accuracy"],
    "macro_f1": summary["overall"]["three_class_report"]["macro avg"]["f1-score"],
    "attack_success_recall": summary["overall"]["three_class_report"]["attack_success"]["recall"],
    "localization_f1": summary["overall"]["localization"]["component_micro_f1"],
    "summary": str(output / "V22_ALL_BERT_TEST_SUMMARY.json"),
    "comparison_json": str(output / "V22_ALL_BERT_VALIDATION_TEST_COMPARISON.json"),
    "comparison_markdown": str(output / "V22_ALL_BERT_VALIDATION_TEST_COMPARISON.md"),
}, ensure_ascii=False, indent=2))
PY
