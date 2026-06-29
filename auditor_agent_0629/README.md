# auditor_agent_0629

ACI-style attack evaluation on MARBLE / MultiAgentBench.

## Benchmark

- Scenarios: Research, Coding, Minecraft, Bargaining
- Samples: 2 per scenario
- Topologies: graph, star, tree
- Attacks: 5 surfaces x 3 objectives
- Scale: 24 clean + 360 attacked = 384 runs

Werewolf is not included in this stable runner because the local MARBLE checkout does not provide a matching `multiagentbench/werewolf_main.jsonl` adapter.

## Results

```text
output/benchmark384_latest/
  merged/
  judge_output_strict/
  analysis_output_strict/
```

Current audited ASR:

```text
marker-delta:       30 / 360 = 8.33%
semantic-confirmed: 24 / 360 = 6.67%
confirmed + likely: 25 / 360 = 6.94%
```

## Run

```bash
cd auditor_agent_marbel
pip install -e .

export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_API_BASE="https://your-endpoint/v1"
export OPENAI_BASE_URL="https://your-endpoint/v1"

python scripts/run_benchmark384_pooled.py \
  --marble-root /path/to/MARBLE \
  --run-config configs/benchmark_384_multiscenario.yaml \
  --attack-spec data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl \
  --output-root ../benchmark384_multiscenario_runs
```

See the repository root README for judge commands.
