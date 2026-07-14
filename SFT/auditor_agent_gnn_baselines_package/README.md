# V12 SFT Auditor vs GNN Baselines

这个文件夹用于复现当前主线对比实验：`V12-SFT auditor` vs `G-Safeguard` / `BlindGuard`。

## 版本说明

- 主线 SFT 版本：`../auditor_agent_sft_v12_package`
- GNN 对比代码：`server_scripts/`
- 最新 common50 结果：`results/common50_latest/`
- 会议展示简版：`MEETING_BRIEF.md`
- 已归档探索版：`../_archive_experimental_variants/`

`V12` 是轻量 graph-grounded SFT auditor，不是 GNN。它输出完整 audit JSON，包括 `verdict`、`surface/objective`、`localization` 和 evidence trace。

## 对比范围

公平比较：

- run-level safe/unsafe
- safe F1 / unsafe F1
- agent-level localization projection

不强行比较，写 `N/A`：

- GNN 的 surface/objective
- GNN 的 edge/tool/global localization
- GNN 的 JSON audit quality

## Agent Label Policies

脚本会同时跑两套：

- `strict_agent`: 只把 `N::agentX` 算 agent label
- `agent_or_tool_owner`: 把 `N::agentX` 和 `T::agentX` 算 agent label

两套都不会把 `E::agentA->agentB` 或 `G::run` 硬算成 agent。

## 一键复现

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor

rm -rf gnn_vs_sft_common50 marble_agent_graph_common50

CUDA_VISIBLE_DEVICES=0 bash Auditor-agent/SFT/auditor_agent_gnn_baselines_package/server_scripts/run_all_gnn_baselines_common50.sh
```

## 输出位置

```text
gnn_vs_sft_common50/strict_agent/comparison_table.json
gnn_vs_sft_common50/agent_or_tool_owner/comparison_table.json
```

仓库内也保存了当前结果：

```text
results/common50_latest/strict_agent/comparison_table.json
results/common50_latest/agent_or_tool_owner/comparison_table.json
```

## 当前结论

- `V8-SFT` 的 run-level 判断最强，但定位弱。
- `V12-SFT` 的 run-level 判断略低于 V8，但 agent-level 定位接近 G-Safeguard，并且保留完整 audit JSON。
- `G-Safeguard` 在 agent ranking 上有优势，但不能输出完整 MAS audit。
- `BlindGuard` 在该 MARBLE audit setting 上失配较明显。
