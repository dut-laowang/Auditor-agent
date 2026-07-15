# MAS Auditor SFT V14-HQ

这一版用于重建高质量 SFT 数据，格式与 V12 对齐，主要改动只在标签质量和脱敏质量。

## 核心思路

- SFT schema 保持：`Graph-grounded-Candidate-SFT/v12`
- 输入不暴露 `attack_id`、`success_marker`、`attack_metadata`、原始 ACI marker、Injected instruction 等字段。
- 标签不再扫描 prompt/config/private metadata；只在可见 `agent/message/tool/final output` 中出现成功证据时标为 `attack_success`。
- 原始 marker 被移除，但保留有害输出语义：
  - disclosure: `[REDACTED_PRIVATE_VALUE]`
  - disruption: `[TASK_INTERRUPTION_OUTPUT]`
  - hijacking: `[TASK_REPLACEMENT_OUTPUT]`

## 当前数据

路径：`sft_dataset_graph_grounded_v14_hq/`

```text
total: 7362
train: 5879
test:  1483

all:
  clean_safe:      192
  attack_failed:  4917
  attack_success: 2253

test:
  clean_safe:      48
  attack_failed:  983
  attack_success: 452
```

覆盖：

```text
scenario: bargaining / coding / minecraft / research
topology: chain / graph / star / tree
surface:  input / memory / message / profile / tool / dual / none
objective: disclosure / disruption / hijacking / none
```

质量检查：

```text
leak_check: {}
invalid evidence/component refs: 0
attack_success without visible semantic evidence: 0
```

## 重新构建

核心代码：

```text
scripts/build_v14_hq_dataset.py
```

本地重建命令示例：

```powershell
$py="D:\FIRST_COPILOT\.miniforge\envs\auditor-marble\python.exe"
$pkg="D:\FIRST_COPILOT\plan_e\auditor_agent\SFT\auditor_agent_sft_v14_hq_label_package"

& $py "$pkg\scripts\build_v14_hq_dataset.py" `
  --zip "D:\FIRST_COPILOT\plan_e\benchmark384_multiscenario_runs(1).zip" `
        "D:\FIRST_COPILOT\plan_e\benchmark384_multiscenario_runs_6_9.zip" `
        "D:\FIRST_COPILOT\plan_e\benchmark384_task12_noncanonical_runs.zip" `
  --output-dir "$pkg\sft_dataset_graph_grounded_v14_hq"
```

输出文件：

```text
all.jsonl
train.jsonl
test.jsonl
stats.json
manual_quality_sample_50_v14_hq.json
```

## 训练评测

`server_scripts/` 保留 Qwen3-8B LoRA 训练与评测脚本，数据目录直接指向：

```text
sft_dataset_graph_grounded_v14_hq
```
