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
git clone https://github.com/dut-laowang/Auditor-agent.git
git clone https://github.com/ulab-uiuc/MARBLE.git

cd Auditor-agent/auditor_agent_0629/auditor_agent_marbel
pip install -e .

export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_API_BASE="https://your-endpoint/v1"
export OPENAI_BASE_URL="https://your-endpoint/v1"

python scripts/run_benchmark384_pooled.py \
  --marble-root ../../../MARBLE \
  --run-config configs/benchmark_384_multiscenario.yaml \
  --attack-spec data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl \
  --output-root ../benchmark384_multiscenario_runs
```

Judge and summarize:

```bash
python scripts/judge_attack_success.py \
  --merged-dir ../benchmark384_multiscenario_runs/final/merged \
  --output-dir ../benchmark384_multiscenario_runs/final/_judge_tmp

python scripts/analyze_and_export_sft.py \
  --judge-dir ../benchmark384_multiscenario_runs/final/_judge_tmp \
  --output-dir ../benchmark384_multiscenario_runs/final/analysis_output_strict
```

To change the experiment scope:

- edit `configs/benchmark_384_multiscenario.yaml` for scenarios, sample ids, and topologies
- edit or copy `data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl` for attack surfaces, objectives, placements, and payloads

The pooled runner computes the run count from those two files, so it can run smaller or larger subsets than the published 384-run benchmark.
