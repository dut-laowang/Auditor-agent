#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-$BASE/v19_three_track_results_$RUN_ID}"

if [[ -e "$RESULT_ROOT/FINAL_RUN_COMPLETE" ]]; then
  echo "Refusing to rerun a completed final experiment: $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT/summary" "$RESULT_ROOT/logs"
printf '%s\n' "$RESULT_ROOT" > "$RESULT_ROOT/RESULT_ROOT_PATH.txt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

for track in marble_only autogen_only mixed; do
  data="$PKG/three_track_datasets/$track"
  track_root="$RESULT_ROOT/$track"
  adapter="$track_root/adapter"
  validation="$track_root/validation"
  final_test="$track_root/final_test"
  mkdir -p "$track_root"

  python "$PKG/scripts/restore_track_data.py" "$data" \
    2>&1 | tee "$track_root/data_restore.log"
  python "$PKG/scripts/audit_v19_integrity.py" \
    --data-dir "$data" --output "$track_root/pretraining_quality_audit.json" \
    2>&1 | tee "$track_root/pretraining_quality_audit.log"
  python "$PKG/scripts/audit_lexical_shortcuts.py" \
    --train-file "$data/train.jsonl" \
    --validation-file "$data/validation.jsonl" \
    --output "$track_root/lexical_shortcut_validation.json"
  counterfactual_data="$track_root/validation_counterfactuals"
  if [[ ! -d "$counterfactual_data" ]]; then
    python "$PKG/scripts/make_validation_ablations.py" \
      --validation-file "$data/validation.jsonl" \
      --shortcut-report "$track_root/lexical_shortcut_validation.json" \
      --output-dir "$counterfactual_data" --seed 42
  fi

  TRACK="$track" OUT="$adapter" BASE="$BASE" REPO="$REPO" GPU="$GPU" \
    bash "$PKG/server_scripts/run_train_v19.sh" \
    2>&1 | tee "$track_root/train_driver.log"

  # Validate the clean set and every predetermined counterfactual. No model or
  # hyperparameter choice is changed after these fixed evaluations.
  EVAL="$PKG/server_scripts/eval_qwen3_fullschema_v19.py"
  if [[ ! -f "$validation/clean/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL" \
      --mode sft --model Qwen/Qwen3-8B --adapter "$adapter" \
      --test-file "$data/validation.jsonl" --dataset-role validation \
      --output-dir "$validation/clean" --max-new-tokens 1024 \
      --batch-size "$EVAL_BATCH_SIZE" --resume
  fi
  for file in "$counterfactual_data"/*.jsonl; do
    name="$(basename "$file" .jsonl)"
    if [[ ! -f "$validation/$name/metrics.json" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL" \
        --mode sft --model Qwen/Qwen3-8B --adapter "$adapter" \
        --test-file "$file" --dataset-role validation \
        --output-dir "$validation/$name" --max-new-tokens 1024 \
        --batch-size "$EVAL_BATCH_SIZE" --resume
    fi
  done
  python "$PKG/scripts/summarize_counterfactuals.py" \
    --result-root "$validation" \
    --output "$validation/counterfactual_summary.json"

  # The protocol is fixed above; consume this track's final test exactly once.
  TRACK="$track" ADAPTER="$adapter" OUTPUT="$final_test" PKG="$PKG" GPU="$GPU" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    bash "$PKG/server_scripts/run_final_test_once_v19.sh" \
    2>&1 | tee "$track_root/final_test_driver.log"
done

python "$PKG/scripts/aggregate_three_track_metrics.py" \
  --marble "$RESULT_ROOT/marble_only/final_test/metrics.json" \
  --autogen "$RESULT_ROOT/autogen_only/final_test/metrics.json" \
  --mixed "$RESULT_ROOT/mixed/final_test/metrics.json" \
  --output "$RESULT_ROOT/summary/three_track_average.json"

python "$PKG/scripts/write_sha256_manifest.py" "$RESULT_ROOT"
touch "$RESULT_ROOT/FINAL_RUN_COMPLETE"
tar -czf "$RESULT_ROOT.tar.gz" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"

echo "All V19 experiments completed."
echo "Result folder: $RESULT_ROOT"
echo "Transfer archive: $RESULT_ROOT.tar.gz"
