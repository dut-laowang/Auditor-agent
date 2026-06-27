# Auditor Agent MARBLE

Graph-aware attack runner for MARBLE / MultiAgentBench. This module keeps MARBLE as the upstream benchmark and adds an ACI-inspired attack layer for collecting attacked multi-agent trajectories.

Upstream references:

- MARBLE: https://github.com/ulab-uiuc/MARBLE
- MultiAgentBench paper: https://arxiv.org/abs/2503.01935
- ACIArena paper: https://arxiv.org/abs/2604.07775

## What This Does

The runner uses official MARBLE JSONL samples as clean tasks, then injects attacks at MAS-specific surfaces:

1. `input`: task / user input modification.
2. `profile`: target agent profile replacement or append.
3. `message`: source-to-target communication edge poisoning.
4. `memory`: target agent memory poisoning.
5. `tool`: action/tool description or result poisoning.

Each attack has one objective:

- `disclosure`
- `disruption`
- `hijacking`

The first pilot scope is intentionally small:

- Research: 3 samples
- Coding: 3 samples
- Database: 3 samples
- Topologies: `graph`, `star`, `tree`

This is enough to validate the graph-aware attack design before scaling to all 100 samples per scenario.

## Directory

```text
auditor_agent_marbel/
  configs/
  data/
    attack_specs/
    sample_manifest/
  scripts/
  src/auditor_agent_marbel/
```

## Server Quick Start

The intended server project root is:

```bash
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
```

Clone both repos side by side:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
git clone https://github.com/dut-laowang/Auditor-agent.git
git clone https://github.com/ulab-uiuc/MARBLE.git
cd Auditor-agent/auditor_agent_marbel
```

Create environment:

```bash
conda create -n auditor-marble python=3.10 -y
conda activate auditor-marble
pip install -e .
cd ../../MARBLE
pip install -e .
cd ../Auditor-agent/auditor_agent_marbel
```

Dry-run the pilot without calling LLMs:

```bash
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --attack-spec data/attack_specs/pilot_attack_specs.jsonl \
  --output-dir outputs/pilot_dry_run \
  --dry-run
```

Run clean baselines:

```bash
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --output-dir outputs/pilot_clean \
  --clean-only
```

Run attacked pilot:

```bash
python -m auditor_agent_marbel.runner.run_attack \
  --marble-root /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/MARBLE \
  --run-config configs/pilot_run.yaml \
  --attack-spec data/attack_specs/pilot_attack_specs.jsonl \
  --output-dir outputs/pilot_attacked
```

## Outputs

Each run creates:

```text
outputs/<run_name>/
  run_manifest.jsonl
  configs/
  trajectories/
```

`trajectory.jsonl` records attack placement, selected sample, injected content, messages, memory writes, tool calls, and run status.

## Design Notes

This code does not fork MARBLE internals. It imports official MARBLE modules from `--marble-root` and installs runtime hooks around:

- `BaseAgent.act`
- `BaseAgent.send_message`
- `BaseAgent.receive_message`
- `BaseEnvironment.apply_action`

That keeps benchmark tasks and execution semantics aligned with the original MARBLE code while allowing graph-aware attack instrumentation.
