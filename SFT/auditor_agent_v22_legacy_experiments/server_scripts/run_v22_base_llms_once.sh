#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"
REPO="${REPO:-$BASE/Auditor-agent}"
RUN="${V22_LEGACY_RUN:-$BASE/v22_all_run}"
OUT="${V22_SUPPLEMENT_RUN:-$BASE/v22_legacy_supplement_run}/baselines"
EVAL="$REPO/SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/eval_qwen3_fullschema_v19.py"
TEST="$RUN/modernbert_sealed_test_source/test.jsonl"
GPU="${GPU:-0}"
mkdir -p "$OUT"
run_one() {
  local model="$1" revision="$2" name="$3" batch="$4" target="$OUT/$3"
  if [[ -f "$target/metrics.json" ]]; then echo "SKIP completed $name"; return; fi
  CUDA_VISIBLE_DEVICES="$GPU" python "$EVAL" --mode base --model "$model" --revision "$revision" \
    --test-file "$TEST" --dataset-role test --sealed-test-ack FINAL_ONCE --output-dir "$target" \
    --max-input-len 8192 --max-new-tokens 1400 --batch-size "$batch" --resume --disable-cudnn-sdp
}
run_one Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3_8b_base "${QWEN8_BASE_BATCH:-4}"
run_one Qwen/Qwen3-32B 9216db5781bf9f97bad3d229c117be829f1811ca qwen3_32b_base "${QWEN32_BASE_BATCH:-1}"
