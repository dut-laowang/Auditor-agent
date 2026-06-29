# Auditor Agent MARBLE

Runner, attack specs, and analysis scripts for ACI-on-MARBLE benchmarks.

Important files:

```text
configs/benchmark_384_multiscenario.yaml
data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl
scripts/run_benchmark384_pooled.py
scripts/judge_attack_success.py
scripts/analyze_and_export_sft.py
src/auditor_agent_marbel/
```

Minimal run:

```bash
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

To change the experiment scope:

- edit `configs/benchmark_384_multiscenario.yaml` for scenarios, sample ids, and topologies
- edit or copy `data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl` for attack surfaces, objectives, placements, and payloads

The pooled runner computes the run count from those two files, so it can run smaller or larger subsets than the published 384-run benchmark.
