#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
MODEL_ROOT="${MODEL_ROOT:-$BASE/sft_models}"
GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PKG="$REPO/SFT/auditor_agent_sft_v22_appworld_package"
V22_RESULTS="${V22_RESULTS:-$BASE/v22_appworld_marble_validation}"
SOURCE="$V22_RESULTS/dataset/audit_sft"
OUT="${V22_TEACHER_RESULTS:-$BASE/v22_qwen32b_teacher_expansion}"
TEACHER="$OUT/qwen32b_enrichment.jsonl"
EXPANDED="$OUT/expanded_audit_sft"

cd "$REPO"
mkdir -p "$OUT"
test -f "$SOURCE/train.jsonl"
test -f "$SOURCE/validation.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" python "$PKG/server_scripts/qwen32b_enrich_v22_reports.py" \
  --train-file "$SOURCE/train.jsonl" --output "$TEACHER" \
  --model Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --batch-size "${TEACHER_BATCH:-2}" --max-input-len 8192 --max-new-tokens 384

python "$PKG/scripts/merge_qwen32b_enrichment.py" \
  --v22-data "$SOURCE" --teacher-output "$TEACHER" --output-dir "$EXPANDED"

tar -czf "$BASE/v22_qwen32b_teacher_expansion.tar.gz" -C "$BASE" "$(basename "$OUT")"
echo "DONE: $BASE/v22_qwen32b_teacher_expansion.tar.gz"
