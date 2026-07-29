#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
PKG="$REPO/SFT/auditor_agent_sft_v17_semantic_package"
DATA="$PKG/sft_dataset_graph_grounded_v17_semantic"
OUT="${OUT:-$BASE/sft_models/qwen3-8b-mas-auditor-lora-v17-semantic-e2-b4g4}"
GPU="${GPU:-0}"

SUBSET_DIR="$BASE/qwen3_8b_sft_v17_semantic_subsets"
SUBSET50="$SUBSET_DIR/v17_semantic_test50.jsonl"
SUBSET200="$SUBSET_DIR/v17_semantic_test200.jsonl"
EVAL50="$BASE/qwen3_8b_sft_v17_semantic_eval50"
EVAL200="$BASE/qwen3_8b_sft_v17_semantic_eval200"
EVALFULL="$BASE/qwen3_8b_sft_v17_semantic_eval_full"
EVALCOMMON="$BASE/qwen3_8b_sft_v17_semantic_eval_v12_common50"
COMMON50="${COMMON50:-$PKG/comparison_sets/v12_common50.jsonl}"
V12_COMMON50_METRICS="${V12_COMMON50_METRICS:-$PKG/comparison_sets/v12_common50_metrics.json}"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1

mkdir -p "$SUBSET_DIR"

test -f "$DATA/train.jsonl"
test -f "$DATA/test.jsonl"
test -f "$DATA/all.jsonl"
if [[ -f "$DATA/SHA256SUMS" ]]; then
  (cd "$DATA" && sha256sum -c SHA256SUMS)
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/train_qwen3_lora_sft.py" \
  --model Qwen/Qwen3-8B \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 4 \
  --grad-accum 4

python "$PKG/server_scripts/make_stratified_subset.py" \
  --input-file "$DATA/test.jsonl" \
  --output-file "$SUBSET50" \
  --n 50 \
  --seed 42

python "$PKG/server_scripts/make_stratified_subset.py" \
  --input-file "$DATA/test.jsonl" \
  --output-file "$SUBSET200" \
  --n 200 \
  --seed 42

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$OUT" \
  --test-file "$SUBSET50" \
  --output-dir "$EVAL50" \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$OUT" \
  --test-file "$SUBSET200" \
  --output-dir "$EVAL200" \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$OUT" \
  --test-file "$DATA/test.jsonl" \
  --output-dir "$EVALFULL" \
  --max-new-tokens 1024

if [[ -n "${COMMON50:-}" && -f "$COMMON50" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
    --mode sft \
    --model Qwen/Qwen3-8B \
    --adapter "$OUT" \
    --test-file "$COMMON50" \
    --output-dir "$EVALCOMMON" \
    --max-new-tokens 1024

  if [[ -n "${V12_COMMON50_METRICS:-}" && -f "$V12_COMMON50_METRICS" ]]; then
    python "$PKG/server_scripts/compare_eval_metrics.py" \
      --before "$V12_COMMON50_METRICS" \
      --after "$EVALCOMMON/metrics.json" \
      --output "$EVALCOMMON/v12_vs_v17.json"
  fi
fi
