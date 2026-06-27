#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/gs/bs/tgh-26IAW/hongbo/project_4_coauthor}"

cd "${BASE_DIR}"
if [ ! -d Auditor-agent ]; then
  git clone https://github.com/dut-laowang/Auditor-agent.git
fi
if [ ! -d MARBLE ]; then
  git clone https://github.com/ulab-uiuc/MARBLE.git
fi

cd Auditor-agent/auditor_agent_marbel
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root "${BASE_DIR}/MARBLE" \
  --run-config configs/pilot_run.yaml \
  --attack-spec data/attack_specs/pilot_attack_specs.jsonl \
  --output-dir outputs/pilot_dry_run \
  --dry-run

