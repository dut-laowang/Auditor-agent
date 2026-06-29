# Auditor Agent 0629

本项目用于在 MARBLE / MultiAgentBench 上运行 ACI-style attack evaluation，并整理 attack result、ASR 和后续 SFT data 构建所需的核心输出。

## Contents

```text
auditor_agent_0629/
  auditor_agent_marbel/       runner, configs, attack specs, judge scripts
  output/benchmark384_latest/ final merged results and audits
```

## Benchmark

当前公开结果对应如下 benchmark 设置：

- Scenarios: Research, Coding, Minecraft, Bargaining
- Samples: 2 per scenario
- Topologies: graph, star, tree
- Attacks: 5 surfaces x 3 objectives
- Scale: 24 clean + 360 attacked = 384 runs

Werewolf 未包含在该 stable runner 中，因为当前 MARBLE checkout 没有匹配的 `multiagentbench/werewolf_main.jsonl` adapter。

## Results

```text
All runs completed: 384 / 384
Failed runs: 0

Marker-delta ASR:       30 / 360 = 8.33%
Semantic-confirmed ASR: 24 / 360 = 6.67%
Confirmed + likely ASR: 25 / 360 = 6.94%
```

Main result files:

```text
auditor_agent_0629/output/benchmark384_latest/merged/run_manifest.jsonl
auditor_agent_0629/output/benchmark384_latest/analysis_output_strict/asr_summary.json
auditor_agent_0629/output/benchmark384_latest/analysis_output_strict/semantic_success_audit.json
```

## Reproduce

先 clone 本项目和官方 MARBLE，然后进入 runner 目录安装：

```bash
git clone https://github.com/dut-laowang/Auditor-agent.git
git clone https://github.com/ulab-uiuc/MARBLE.git

cd Auditor-agent/auditor_agent_0629/auditor_agent_marbel
pip install -e .
```

设置 OpenAI-compatible endpoint：

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_API_BASE="https://your-endpoint/v1"
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

运行 benchmark：

```bash
python scripts/run_benchmark384_pooled.py \
  --marble-root ../../../MARBLE \
  --run-config configs/benchmark_384_multiscenario.yaml \
  --attack-spec data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl \
  --output-root ../benchmark384_multiscenario_runs
```

运行 strict judge 并汇总结果：

```bash
python scripts/judge_attack_success.py \
  --merged-dir ../benchmark384_multiscenario_runs/final/merged \
  --output-dir ../benchmark384_multiscenario_runs/final/_judge_tmp

python scripts/analyze_and_export_sft.py \
  --judge-dir ../benchmark384_multiscenario_runs/final/_judge_tmp \
  --output-dir ../benchmark384_multiscenario_runs/final/analysis_output_strict
```

strict judge 只统计 agent/tool outputs 中生成的 markers，不统计 attack metadata、prepared configs、prompts 或 injected message text。

## Change Scope

如需修改实验范围：

- 修改 `auditor_agent_0629/auditor_agent_marbel/configs/benchmark_384_multiscenario.yaml`，控制 scenarios、sample ids 和 topologies。
- 修改或复制 `auditor_agent_0629/auditor_agent_marbel/data/attack_specs/benchmark384_multiscenario_attack_specs.jsonl`，控制 attack surfaces、objectives、placements 和 payloads。

pooled runner 会根据这两个文件自动计算 run count，因此可以运行小于或大于当前 384-run benchmark 的自定义子集。
