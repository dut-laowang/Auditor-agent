#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$ROOT/model_config.yaml"
JUDGE="$ROOT/judge_config.yaml"

run_one () {
  local MAS="$1"
  local AGENT="$2"
  local OUT="$ROOT/outputs/${MAS}_code_disclosure_10/raw_runs"

  echo "=============================="
  echo "Running MAS=$MAS malicious_agent=$AGENT"
  echo "Output: $OUT"
  echo "=============================="

  rm -rf "$OUT"

  python "$ROOT/run_single_attack.py" \
    --mas "$MAS" \
    --suite disclosure \
    --task-domain code \
    --all-attacks \
    --malicious-agents "$AGENT" \
    --task-limit 2 \
    --defense none \
    --model-config "$MODEL" \
    --judge-config "$JUDGE" \
    --output-dir "$OUT"

  if [ -f "$ROOT/check_outputs.py" ]; then
    python "$ROOT/check_outputs.py" \
      --input-dir "$OUT" \
      --expected 10 || true
  fi
}

run_one autogen assistant
run_one camel assistant
run_one mad negative
run_one agentverse solver
run_one llm_debate debater_0
run_one metagpt qa_engineer
run_one sc sc1
