#!/usr/bin/env bash
set -euo pipefail

: "${PKG:?Set PKG to the V19 package path}"
: "${ADAPTER:?Set ADAPTER to the frozen track adapter path}"
: "${OUTPUT:?Set OUTPUT to a validation result directory}"
: "${TRACK:?Set TRACK to marble_only, autogen_only, or mixed}"
case "$TRACK" in marble_only|autogen_only|mixed) ;; *) echo "Invalid TRACK=$TRACK" >&2; exit 2;; esac
GPU="${GPU:-0}"
DATA="$PKG/three_track_datasets/$TRACK"
EVAL="$PKG/server_scripts/eval_qwen3_fullschema_v19.py"

run_eval() {
  local name="$1"
  local file="$2"
  CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL" \
    --mode sft --model Qwen/Qwen3-8B --adapter "$ADAPTER" \
    --test-file "$file" --dataset-role validation \
    --output-dir "$OUTPUT/$name" --max-new-tokens 1024 --resume
}

run_eval clean "$DATA/validation.jsonl"
for file in "$DATA"/validation_counterfactuals/*.jsonl; do
  name="$(basename "$file" .jsonl)"
  run_eval "$name" "$file"
done

python "$PKG/scripts/summarize_counterfactuals.py" \
  --result-root "$OUTPUT" --output "$OUTPUT/counterfactual_summary.json"
