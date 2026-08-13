#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
V20="$REPO/SFT/auditor_agent_sft_v20_autogen_package"
V19="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package"
GNN="$REPO/SFT/auditor_agent_gnn_baselines_package"
RAW_DATA="${V20_DATA:-$V20/dataset_bundle}"
RESULTS="${RESULTS:-$BASE/v20_autogen_core_validation_v2}"
DATA="$RESULTS/context_filtered_dataset"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
GPU="${GPU:-0}"

mkdir -p "$RESULTS" "$MODEL_ROOT"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python "$V19/scripts/restore_track_data.py" "$RAW_DATA"

python "$V19/scripts/audit_v19_integrity.py" --data-dir "$RAW_DATA" --output "$RESULTS/data_integrity.json"
python "$V20/scripts/audit_autogen_observable_v2.py" \
  --data-dir "$RAW_DATA" --output "$RESULTS/autogen_observable_quality.json"
python "$V20/scripts/filter_qwen_context_v20.py" \
  --input-dir "$RAW_DATA" --output-dir "$DATA" --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --max-len 8192
python "$V19/scripts/audit_lexical_shortcuts.py" \
  --train-file "$DATA/train.jsonl" --validation-file "$DATA/validation.jsonl" \
  --output "$RESULTS/lexical_shortcut_validation.json"

# TF-IDF validation baseline.
python "$V20/scripts/tfidf_v20.py" --data-dir "$DATA" --output "$RESULTS/tfidf/metrics.json"

# Core Qwen3-8B SFT, with the exact V19 training settings and cached base model.
QWEN_ADAPTER="$MODEL_ROOT/qwen3-8b-mas-auditor-lora-v20-autogen-observable-v2"
if [[ ! -f "$QWEN_ADAPTER/TRAINING_COMPLETE" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/train_qwen3_lora_sft_v19.py" \
    --model Qwen/Qwen3-8B --data-dir "$DATA" --output-dir "$QWEN_ADAPTER" \
    --max-len 8192 \
    --epochs 2 --lr 2e-4 --batch 2 --grad-accum 8 --seed 42 --resume auto \
    2>&1 | tee "$RESULTS/qwen3_8b_training.log"
  touch "$QWEN_ADAPTER/TRAINING_COMPLETE"
fi
CUDA_VISIBLE_DEVICES="$GPU" python "$V19/server_scripts/eval_qwen3_fullschema_v19.py" \
  --mode sft --model Qwen/Qwen3-8B --adapter "$QWEN_ADAPTER" \
  --test-file "$DATA/validation.jsonl" --dataset-role validation \
  --output-dir "$RESULTS/qwen3_8b_clean" --max-new-tokens 1024 \
  --batch-size "${QWEN_EVAL_BATCH:-4}" --resume

# ModernBERT-8192, same revision and V19 hyperparameters; model downloads reuse HF_HOME.
MODERN_OUT="$MODEL_ROOT/modernbert-base-8192-sdpa-fp32-multitask-v20-autogen-observable-v2"
V20_TRAIN_SHA="$(sha256sum "$DATA/train.jsonl" | awk '{print $1}')"
V20_VALIDATION_SHA="$(sha256sum "$DATA/validation.jsonl" | awk '{print $1}')"
mkdir -p "$MODERN_OUT" "$RESULTS/modernbert"
if [[ ! -f "$MODERN_OUT/TRAINING_COMPLETE.json" ]]; then
  python "$V19/server_scripts/modernbert_multitask_v19.py" \
    --mode train --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
    --data-file "$DATA/train.jsonl" --dataset-role train --output-dir "$MODERN_OUT" \
    --expected-train-sha256 "$V20_TRAIN_SHA" --expected-validation-sha256 "$V20_VALIDATION_SHA" \
    --max-len 8192 --attn-implementation sdpa --input-mode user --epochs 3 --lr 2e-5 \
    --batch "${MODERN_TRAIN_BATCH:-2}" --grad-accum "${MODERN_GRAD_ACCUM:-8}" \
    --lambda-scope 1.0 --lambda-component 1.0 --seed 42 \
    2>&1 | tee "$RESULTS/modernbert_training.log"
fi
python "$V19/server_scripts/modernbert_multitask_v19.py" \
  --mode eval --model answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --checkpoint "$MODERN_OUT/checkpoint-epoch-3.pt" \
  --data-file "$DATA/validation.jsonl" --dataset-role validation \
  --expected-train-sha256 "$V20_TRAIN_SHA" --expected-validation-sha256 "$V20_VALIDATION_SHA" \
  --output-dir "$RESULTS/modernbert" --max-len 8192 --attn-implementation sdpa \
  --input-mode user --batch "${MODERN_EVAL_BATCH:-4}" --seed 42 \
  2>&1 | tee "$RESULTS/modernbert/evaluation.log"

# Supervised graph adaptations: G-Safeguard and TAM encoder only.
REFS="$BASE/gnn_refs"
GSAFE="$REFS/G-safeguard"
BLIND="$REFS/BlindGuard"
test -d "$GSAFE/.git"
test -d "$BLIND/.git"
git -C "$GSAFE" checkout --detach 890c99f1cbc864e9ff0c85859619a14f42bc9cab
git -C "$BLIND" checkout --detach 1889c20a326ba9ba9a6982744d473626e74f9986
GNN_CACHE="$MODEL_ROOT/v20_autogen_observable_v2_gnn_component_cache"
python "$GNN/server_scripts/v19_component_gnn_multitask.py" train-validation \
  --model-kind gat --official-dir "$GSAFE/TA" --data-dir "$DATA" \
  --expected-train-sha256 "$V20_TRAIN_SHA" --expected-validation-sha256 "$V20_VALIDATION_SHA" \
  --cache-dir "$GNN_CACHE" --output-dir "$RESULTS/gsafeguard" \
  --epochs 20 --lr 0.001 --hidden-dim 512 --latent-dim 256 \
  2>&1 | tee "$RESULTS/gsafeguard.log"
python "$GNN/server_scripts/v19_component_gnn_multitask.py" train-validation \
  --model-kind tam --official-dir "$BLIND/MA" --data-dir "$DATA" \
  --expected-train-sha256 "$V20_TRAIN_SHA" --expected-validation-sha256 "$V20_VALIDATION_SHA" \
  --cache-dir "$GNN_CACHE" --output-dir "$RESULTS/tam_encoder" \
  --epochs 20 --lr 0.001 --hidden-dim 512 --latent-dim 256 \
  2>&1 | tee "$RESULTS/tam_encoder.log"

python "$V19/scripts/write_sha256_manifest.py" "$RESULTS"
touch "$RESULTS/VALIDATION_COMPLETE"
tar -czf "$RESULTS.tar.gz" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")"
echo "V20 AutoGen core validation complete; sealed test was not consumed."
echo "Results: $RESULTS"
