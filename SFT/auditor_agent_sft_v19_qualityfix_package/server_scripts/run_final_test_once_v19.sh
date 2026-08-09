#!/usr/bin/env bash
set -euo pipefail

: "${PKG:?Set PKG to the V19 package path}"
: "${ADAPTER:?Set ADAPTER to the frozen V19 adapter path}"
: "${OUTPUT:?Set OUTPUT to a new final-evaluation directory}"
GPU="${GPU:-0}"
TRACK="${TRACK:?Set TRACK to marble_only, autogen_only, or mixed}"
case "$TRACK" in marble_only|autogen_only|mixed) ;; *) echo "Invalid TRACK=$TRACK" >&2; exit 2;; esac
DATA="$PKG/three_track_datasets/$TRACK"

test -f "$ADAPTER/run_manifest.json"
resume_args=()
if [[ -e "$OUTPUT/SEALED_TEST_CONSUMED.json" ]]; then
  resume_args+=(--resume)
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$ADAPTER" \
  --test-file "$DATA/test.jsonl" \
  --dataset-role test \
  --output-dir "$OUTPUT" \
  --max-new-tokens 1024 \
  --sealed-test-ack FINAL_ONCE \
  "${resume_args[@]}"
