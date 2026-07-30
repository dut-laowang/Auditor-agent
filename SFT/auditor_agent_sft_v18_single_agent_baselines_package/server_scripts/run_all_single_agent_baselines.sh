#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
GPU="${GPU:-0}"
PKG="$REPO/SFT/auditor_agent_sft_v18_single_agent_baselines_package"
DATA="$PKG/data"
OUT="${FLAT_OUT:-$BASE/sft_models/qwen3-8b-v18-flat-trajectory-lora-e2-b2g8}"
TRAIN_DONE="$OUT/.v18_flat_training_complete"
SUBSETS="$BASE/qwen3_8b_sft_v18_flat_subsets"
EVAL50="$BASE/qwen3_8b_sft_v18_flat_eval50"
EVAL200="$BASE/qwen3_8b_sft_v18_flat_eval200"
EVALFULL="$BASE/qwen3_8b_sft_v18_flat_eval_full"
ADOG_OFFICIAL="$BASE/agentdog15_v18_eval200_official"
ADOG_ADAPTED="$BASE/agentdog15_v18_eval200_outcome_adapted"
GRAPH200="${GRAPH200:-$BASE/qwen3_8b_sft_v18_observable_eval200/metrics.json}"
SUMMARY="$BASE/v18_single_vs_graph_comparison/comparison_summary.json"

export HF_HOME="${HF_HOME:-$BASE/sft_models/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

test -f "$DATA/train.jsonl.zip"
test -f "$DATA/test.jsonl"
if [[ ! -f "$DATA/train.jsonl" ]]; then
  python -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); z.extract("train.jsonl", sys.argv[2])' \
    "$DATA/train.jsonl.zip" "$DATA"
fi
(cd "$DATA" && sha256sum -c SHA256SUMS)

mkdir -p "$SUBSETS"
python "$PKG/server_scripts/make_stratified_subset.py" \
  --input-file "$DATA/test.jsonl" --output-file "$SUBSETS/test50.jsonl" --n 50 --seed 42
python "$PKG/server_scripts/make_stratified_subset.py" \
  --input-file "$DATA/test.jsonl" --output-file "$SUBSETS/test200.jsonl" --n 200 --seed 42

if [[ -f "$TRAIN_DONE" ]]; then
  echo "Flat training already completed: $TRAIN_DONE"
else
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/train_flat_qwen3_lora.py" \
    --model Qwen/Qwen3-8B \
    --data-dir "$DATA" \
    --output-dir "$OUT" \
    --max-len 6144 \
    --epochs 2 \
    --lr 2e-4 \
    --batch 2 \
    --grad-accum 8 \
    --resume auto
  touch "$TRAIN_DONE"
fi

for spec in \
  "$SUBSETS/test50.jsonl|$EVAL50" \
  "$SUBSETS/test200.jsonl|$EVAL200" \
  "$DATA/test.jsonl|$EVALFULL"; do
  IFS='|' read -r test_file output_dir <<<"$spec"
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_flat_fullschema.py" \
    --mode sft --model Qwen/Qwen3-8B --adapter "$OUT" \
    --test-file "$test_file" --output-dir "$output_dir" \
    --max-new-tokens 1024 --resume
done

AGENTDOG_MODEL="${AGENTDOG_MODEL:-AI45Research/AgentDoG1.5-Llama-3.1-8B}"
for protocol in official_action_safety outcome_adapted; do
  if [[ "$protocol" == "official_action_safety" ]]; then
    output_dir="$ADOG_OFFICIAL"
  else
    output_dir="$ADOG_ADAPTED"
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_agentdog_binary.py" \
    --model "$AGENTDOG_MODEL" \
    --test-file "$SUBSETS/test200.jsonl" \
    --output-dir "$output_dir" \
    --protocol "$protocol" \
    --resume
done

test -f "$GRAPH200"
python "$PKG/server_scripts/compare_baselines.py" \
  --flat "$EVAL200/metrics.json" \
  --graph "$GRAPH200" \
  --agentdog-official "$ADOG_OFFICIAL/metrics.json" \
  --agentdog-adapted "$ADOG_ADAPTED/metrics.json" \
  --output "$SUMMARY"

echo "Completed. Summary: $SUMMARY"
