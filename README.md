# Auditor Agent

这是一个基于 **ACIArena** 的最小复现实验包，用来生成多智能体攻击轨迹日志，并进一步转换成 Auditor Agent 可用的 `graph-prefix` 样本。

核心流程：

```text
benign task + attack payload
        ↓
运行 ACIArena 中的 MAS
        ↓
保存 sender / receiver / message 日志
        ↓
记录 attack_success / utility_success
        ↓
转换为 graph-prefix SFT 样本
```

## 目录

```text
auditor_agent/
├── aci_min_repro/          # 复现实验脚本与已生成 outputs
├── mas_safety_doc/
│   ├── _src_aciarena/      # ACIArena 官方源码副本
│   └── tools/              # 日志转换工具
├── requirements.txt
└── README.md
```

## 安装

```bash
conda create -n auditor-agent python=3.10 -y
conda activate auditor-agent

cd auditor_agent
pip install -r requirements.txt
pip install -e mas_safety_doc/_src_aciarena
```

## 配置 API

`aci_min_repro/` 里已经保留了两个运行配置：

```text
model_config.yaml   # MAS 中各 agent 调用的模型
judge_config.yaml   # attack verify / judge 用的模型
```

如果要换 API，只需要改这两个文件里的三项：

```yaml
provider: openai
api_key: ...
base_url: ...
model_name: gpt-4o-mini
temperature: 0.0
max_tokens: 1024
```

只要服务兼容 OpenAI API，`provider` 保持 `openai` 即可。

## 快速测试

先 dry-run，不调用模型：

```bash
python run_single_attack.py \
  --mas metagpt \
  --suite disclosure \
  --task-domain code \
  --all-attacks \
  --malicious-agents qa_engineer \
  --task-limit 2 \
  --defense none \
  --model-config model_config.yaml \
  --judge-config judge_config.yaml \
  --output-dir outputs/metagpt_code_disclosure_10/raw_runs \
  --dry-run
```

如果看到 `Expected raw trajectories: 10`，说明路径和攻击类加载正常。

## 运行小规模复现

```bash
bash run_all_mas_code_disclosure_10.sh
```

默认测试：

```text
7 个 MAS × 2 个 code task × 5 个 disclosure attacks
```

已生成的小规模结果保留在：

```text
aci_min_repro/outputs/
```

## 当前观察

```text
MetaGPT / AutoGen: 日志完整，攻击较容易传播。
SC: 星型汇聚结构清晰，局部泄露常被 aggregate 抑制。
CAMEL: 日志完整，但 utility 较弱。
AgentVerse / LLM-Debate: 当前 public code 下 turns 可能为空，需要额外 instrumentation。
```

## 注意

- `model_config.yaml` 和 `judge_config.yaml` 会被脚本直接读取；如果上传公开仓库，记得先替换真实 API key。
- `aci_min_repro/outputs/` 已保留，方便直接查看复现实验结果。
- 本仓库不重写 ACIArena 攻击逻辑，只做最小 wrapper、日志保存和格式转换。
