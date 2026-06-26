#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/run_metagpt_code_disclosure_150.py" \
  --model-config "$SCRIPT_DIR/model_config.yaml" \
  --judge-config "$SCRIPT_DIR/judge_config.yaml" \
  --output-dir "$SCRIPT_DIR/outputs/metagpt_code_disclosure_150/raw_runs" \
  --task-limit 30
