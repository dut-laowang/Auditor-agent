# Auditor Agent

ACI-inspired attack evaluation on MARBLE / MultiAgentBench.

## Current Scope

- Scenarios: Research, Coding, Minecraft, Bargaining
- Samples: 2 per scenario
- Topologies: graph, star, tree
- Attacks: 5 surfaces x 3 objectives
- Final scale: 24 clean + 360 attacked = 384 runs

Attack surfaces:

- `input`
- `profile`
- `message`
- `memory`
- `tool`

Attack objectives:

- `disclosure`
- `disruption`
- `hijacking`

## Results

Latest merged output:

```text
output/benchmark384_latest/
  merged/
  judge_output_strict/
  analysis_output_strict/
  run_summary.json
```

Strict output-level ASR:

```text
30 / 360 = 8.33%
```

This strict judge only counts markers produced by agent/tool outputs. It does not count attack payload text merely appearing in configs, prompts, or injected messages.

## Run Locally

```powershell
cd D:\FIRST_COPILOT\plan_e\auditor_agent\auditor_agent_marbel

$env:OPENAI_API_KEY=[Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")
$env:OPENAI_API_BASE=[Environment]::GetEnvironmentVariable("OPENAI_API_BASE","User")
$env:OPENAI_BASE_URL=[Environment]::GetEnvironmentVariable("OPENAI_BASE_URL","User")

D:\FIRST_COPILOT\.miniforge\Scripts\conda.exe run -n auditor-marble python scripts\run_benchmark384_pooled.py `
  --marble-root D:\FIRST_COPILOT\.codex_tmp_marble `
  --run-config configs\benchmark_384_multiscenario.yaml `
  --attack-spec data\attack_specs\benchmark384_multiscenario_attack_specs.jsonl `
  --output-root ..\benchmark384_multiscenario_runs
```

## Layout

```text
auditor_agent_marbel/        runner, attacks, configs, analysis scripts
output/benchmark384_latest/  merged benchmark output only
```
