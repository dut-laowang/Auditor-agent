#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}
REPO=${REPO:-$BASE/Auditor-agent}
PKG=${PKG:-$REPO/SFT/auditor_agent_sft_v12_hq_package}
V12_DATA=${V12_DATA:-$REPO/SFT/auditor_agent_sft_v12_package/sft_dataset_graph_grounded_v12}
HQ_DATA=${HQ_DATA:-$PKG/sft_dataset_graph_grounded_v12_hq}
SFT_OUT=${SFT_OUT:-$BASE/sft_models/qwen3-8b-mas-auditor-lora-v12-hq}

BALANCED50=${BALANCED50:-$BASE/balanced_common_v8_v9_50}
HQ_50=${HQ_50:-$BASE/qwen3_8b_sft_v12_hq_balanced50_eval_tok1024}
HQ_FULL=${HQ_FULL:-$BASE/qwen3_8b_sft_v12_hq_full_eval_tok512}

export HF_HOME=${HF_HOME:-$BASE/sft_models/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$BASE/sft_models/hf_cache/transformers}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$BASE/sft_models/hf_cache/transformers}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

cd "$REPO"
git pull
cd "$BASE"

rm -rf "$SFT_OUT" "$HQ_50" "$HQ_FULL"

python "$PKG/server_scripts/build_v12_hq_dataset.py" \
  --v12-data-dir "$V12_DATA" \
  --output-dir "$HQ_DATA"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$PKG/server_scripts/train_qwen3_lora_sft.py" \
  --model Qwen/Qwen3-8B \
  --data-dir "$HQ_DATA" \
  --output-dir "$SFT_OUT" \
  --max-len 4096 \
  --epochs 2 \
  --lr 2e-4 \
  --batch 1 \
  --grad-accum 16

python "$PKG/server_scripts/make_subset_by_ids.py" \
  --all-file "$HQ_DATA/all.jsonl" \
  --ids-file "$BALANCED50/balanced_common_run_ids.txt" \
  --output-file "$BALANCED50/v12_hq_balanced_common.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$SFT_OUT" \
  --test-file "$BALANCED50/v12_hq_balanced_common.jsonl" \
  --output-dir "$HQ_50" \
  --max-new-tokens 1024

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$PKG/server_scripts/eval_qwen3_fullschema.py" \
  --mode sft \
  --model Qwen/Qwen3-8B \
  --adapter "$SFT_OUT" \
  --test-file "$HQ_DATA/test.jsonl" \
  --output-dir "$HQ_FULL" \
  --max-new-tokens 512

echo "Done."
echo "HQ data: $HQ_DATA"
echo "Balanced50 metrics: $HQ_50/metrics.json"
echo "Full metrics: $HQ_FULL/metrics.json"
