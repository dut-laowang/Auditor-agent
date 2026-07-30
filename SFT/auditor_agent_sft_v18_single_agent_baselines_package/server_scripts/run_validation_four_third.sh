#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
GPU="${GPU:-0}"
PKG="$REPO/SFT/auditor_agent_sft_v18_single_agent_baselines_package"
VDATA="$PKG/validation_data"
FLAT="$VDATA/flat"
GRAPH="$VDATA/graph"
MODEL_QWEN="${MODEL_QWEN:-Qwen/Qwen3-8B}"
MODEL_ADOG="${MODEL_ADOG:-AI45Research/AgentDoG1.5-Llama-3.1-8B}"

QWEN_FLAT_OUT="$BASE/sft_models/qwen3-8b-v18-flat-third-lora"
QWEN_GRAPH_OUT="$BASE/sft_models/qwen3-8b-v18-graph-third-lora"
ADOG_FT_OUT="$BASE/sft_models/agentdog15-llama8b-v18-flat-third-lora"

QWEN_FLAT_EVAL="$BASE/v18_validation_qwen_flat_eval200"
QWEN_GRAPH_EVAL="$BASE/v18_validation_qwen_graph_eval200"
ADOG_FT_EVAL="$BASE/v18_validation_agentdog_finetuned_eval200"
ADOG_OFFICIAL_EVAL="$BASE/v18_validation_agentdog_frozen_official_eval200"
ADOG_ADAPTED_EVAL="$BASE/v18_validation_agentdog_frozen_adapted_eval200"
SUMMARY="$BASE/v18_validation_four_third/comparison_summary.json"

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

for data_dir in "$FLAT" "$GRAPH"; do
  test -f "$data_dir/train.jsonl.zip"
  test -f "$data_dir/test.jsonl"
  if [[ ! -f "$data_dir/train.jsonl" ]]; then
    python -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); z.extract("train.jsonl", sys.argv[2])' \
      "$data_dir/train.jsonl.zip" "$data_dir"
  fi
  (cd "$data_dir" && sha256sum -c SHA256SUMS)
done

train_if_needed() {
  local trainer="$1"
  local model="$2"
  local data_dir="$3"
  local output_dir="$4"
  local marker="$output_dir/.validation_training_complete"
  if [[ -f "$marker" ]]; then
    echo "Training already complete: $output_dir"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python "$trainer" \
    --model "$model" --data-dir "$data_dir" --output-dir "$output_dir" \
    --max-len 6144 --epochs 2 --lr 2e-4 --batch 2 --grad-accum 8 \
    --resume auto
  touch "$marker"
}

evaluate_finetuned() {
  local model="$1"
  local adapter="$2"
  local test_file="$3"
  local output_dir="$4"
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_flat_fullschema.py" \
    --mode sft --model "$model" --adapter "$adapter" \
    --test-file "$test_file" --output-dir "$output_dir" \
    --max-new-tokens 1024 --resume
}

# 1. Qwen trajectory-only, V18 domain trained.
train_if_needed "$PKG/server_scripts/train_flat_qwen3_lora.py" \
  "$MODEL_QWEN" "$FLAT" "$QWEN_FLAT_OUT"
evaluate_finetuned "$MODEL_QWEN" "$QWEN_FLAT_OUT" \
  "$FLAT/test.jsonl" "$QWEN_FLAT_EVAL"

# 2. Qwen graph-grounded/Ours, trained on exactly the same one-third run IDs.
train_if_needed "$REPO/SFT/auditor_agent_sft_v18_observable_package/server_scripts/train_qwen3_lora_sft.py" \
  "$MODEL_QWEN" "$GRAPH" "$QWEN_GRAPH_OUT"
evaluate_finetuned "$MODEL_QWEN" "$QWEN_GRAPH_OUT" \
  "$GRAPH/test.jsonl" "$QWEN_GRAPH_EVAL"

# 3. Official AgentDoG checkpoint further LoRA-tuned on V18-Flat.
train_if_needed "$PKG/server_scripts/train_agentdog_flat_lora.py" \
  "$MODEL_ADOG" "$FLAT" "$ADOG_FT_OUT"
evaluate_finetuned "$MODEL_ADOG" "$ADOG_FT_OUT" \
  "$FLAT/test.jsonl" "$ADOG_FT_EVAL"

# 4. Frozen AgentDoG, official and preregistered outcome-adapted protocols.
for protocol in official_action_safety outcome_adapted; do
  if [[ "$protocol" == "official_action_safety" ]]; then
    output_dir="$ADOG_OFFICIAL_EVAL"
  else
    output_dir="$ADOG_ADAPTED_EVAL"
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/eval_agentdog_binary.py" \
    --model "$MODEL_ADOG" --test-file "$FLAT/test.jsonl" \
    --output-dir "$output_dir" --protocol "$protocol" --resume
done

python "$PKG/server_scripts/compare_validation_four.py" \
  --agentdog-official "$ADOG_OFFICIAL_EVAL/metrics.json" \
  --agentdog-adapted "$ADOG_ADAPTED_EVAL/metrics.json" \
  --agentdog-finetuned "$ADOG_FT_EVAL/metrics.json" \
  --qwen-flat "$QWEN_FLAT_EVAL/metrics.json" \
  --qwen-graph "$QWEN_GRAPH_EVAL/metrics.json" \
  --output "$SUMMARY"

echo "All four validation experiments completed."
echo "Summary: $SUMMARY"
