# GNN Baselines on MARBLE Auditor Data

This package compares V8/V12 SFT auditors with adapted GNN baselines on the same MARBLE balanced test set.

The goal is **not** to force all metrics to match. G-Safeguard and BlindGuard are agent/anomaly detectors, while our SFT auditor outputs a full JSON audit verdict. Therefore unsupported metrics are reported as `N/A`.

## Baselines

- `V8-SFT`
- `V12-SFT`
- `G-Safeguard-style GAT`
- `BlindGuard-style TAM anomaly detector`

## Strict Scope

Comparable:

- binary safe/unsafe accuracy
- safe F1
- unsafe F1
- run AUROC when scores exist
- agent-level localization F1 / top-k hit

Not directly comparable for GNN:

- surface F1
- objective F1
- edge/tool/global localization
- JSON audit quality
- evidence trace quality

These are written as `N/A` for GNN rows.

## Agent Label Policies

The one-shot script reports two GNN adaptations:

- `strict_agent`: only `N::agentX` is converted into an agent label.
- `agent_or_tool_owner`: `N::agentX` and `T::agentX` are converted into agent labels.

In both policies, `E::agentA->agentB` and `G::run` are **not** converted into agent labels. Edge/global localization is reported as `N/A` for GNN rows because these baselines natively rank agents, not edges/tools/global components.

## Minimal Modification Policy

The official GNN repos are not edited. The one-shot script clones:

```text
https://github.com/wslong20/G-safeguard.git
https://github.com/MR9812/BlindGuard.git
```

Our code only adapts MARBLE/V12 SFT rows into the agent graph format expected by those methods and calls official model classes from their `TA/` directories.

## One-shot Run

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor
CUDA_VISIBLE_DEVICES=0 bash Auditor-agent/SFT/auditor_agent_gnn_baselines_package/server_scripts/run_all_gnn_baselines_common50.sh
```

Outputs:

```text
$BASE/gnn_vs_sft_common50/strict_agent/gsafeguard/metrics.json
$BASE/gnn_vs_sft_common50/strict_agent/blindguard/metrics.json
$BASE/gnn_vs_sft_common50/strict_agent/comparison_table.json
$BASE/gnn_vs_sft_common50/agent_or_tool_owner/gsafeguard/metrics.json
$BASE/gnn_vs_sft_common50/agent_or_tool_owner/blindguard/metrics.json
$BASE/gnn_vs_sft_common50/agent_or_tool_owner/comparison_table.json
```

If `torch-geometric` / `torch-scatter` installation fails on the server, the blocker is dependency installation, not the benchmark design.
