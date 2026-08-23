#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
RUNNER="$REPO/SFT/auditor_agent_v22_legacy_experiments/server_scripts/run_v22_closed_llm_colleague_once.sh"
ROOT_OUT="${OUTPUT_DIR:-$PWD/v22_closed_llms_full_test}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp "OpenAI API Key: " OPENAI_API_KEY; echo
  export OPENAI_API_KEY
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  read -rsp "Anthropic API Key: " ANTHROPIC_API_KEY; echo
  export ANTHROPIC_API_KEY
fi
[[ -n "$OPENAI_API_KEY" && -n "$ANTHROPIC_API_KEY" ]] || { echo "Both API keys are required" >&2; exit 1; }

if ! python -c 'import openai, anthropic, sklearn, tqdm' >/dev/null 2>&1; then
  echo "Installing missing API/evaluation dependencies into the active Python environment..."
  python -m pip install -U openai anthropic scikit-learn tqdm
fi

mkdir -p "$ROOT_OUT/logs"
echo "Starting two label-blind V22 evaluations (2,539 identical requests each)."
echo "OpenAI and Anthropic run concurrently; follow logs with:"
echo "  tail -f $ROOT_OUT/logs/openai.log"
echo "  tail -f $ROOT_OUT/logs/anthropic.log"

(
  PROVIDER=openai \
  MODEL="${OPENAI_MODEL:-gpt-4.1-2025-04-14}" \
  API_WORKERS="${OPENAI_WORKERS:-16}" \
  OUTPUT_DIR="$ROOT_OUT/openai" \
  bash "$RUNNER"
) 2>&1 | tee "$ROOT_OUT/logs/openai.log" & OPENAI_PID=$!

(
  PROVIDER=anthropic \
  MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}" \
  API_WORKERS="${ANTHROPIC_WORKERS:-8}" \
  OUTPUT_DIR="$ROOT_OUT/anthropic" \
  bash "$RUNNER"
) 2>&1 | tee "$ROOT_OUT/logs/anthropic.log" & ANTHROPIC_PID=$!

OPENAI_STATUS=0; ANTHROPIC_STATUS=0
wait "$OPENAI_PID" || OPENAI_STATUS=$?
wait "$ANTHROPIC_PID" || ANTHROPIC_STATUS=$?
if (( OPENAI_STATUS != 0 || ANTHROPIC_STATUS != 0 )); then
  echo "Batch incomplete: openai_status=$OPENAI_STATUS anthropic_status=$ANTHROPIC_STATUS" >&2
  echo "Rerun the same command to resume completed rows." >&2
  exit 1
fi

python - "$ROOT_OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
result={"version":"V22-closed-LLMs-batch-complete-v1","status":"PASS","models":{}}
for provider in ("openai","anthropic"):
    p=root/provider/"INFERENCE_COMPLETE.json"
    raw=next((root/provider).glob("V22_*_RAW_RESULTS.tar.gz"))
    result["models"][provider]={"completion":json.loads(p.read_text()),"raw_archive":raw.name,
        "raw_archive_sha256":hashlib.sha256(raw.read_bytes()).hexdigest()}
(root/"BATCH_COMPLETE.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
PY

tar -czf "$ROOT_OUT/V22_CLOSED_LLMS_RAW_RESULTS.tar.gz" -C "$ROOT_OUT" \
  BATCH_COMPLETE.json openai/V22_openai_*_RAW_RESULTS.tar.gz anthropic/V22_anthropic_*_RAW_RESULTS.tar.gz
echo "ALL DONE. RETURN THIS FILE: $ROOT_OUT/V22_CLOSED_LLMS_RAW_RESULTS.tar.gz"
