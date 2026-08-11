# V19 MARBLE component-level GNN baselines

These baselines replace the historical common-50 agent projection with the
native V19 task contract.

## Fair-comparison contract

- train: the frozen 4,565-row V19 MARBLE train split;
- model/threshold selection: the frozen 1,791-row validation split;
- final test: the frozen 1,491-row sealed split, invoked separately;
- input: only the V19 system-visible user record; metadata and assistant targets
  are used solely for supervision and metrics;
- verdict: clean-safe / attack-failed / attack-success;
- localization: multi-label prediction over every supplied `G::`, `N::`, `E::`,
  and `T::` candidate;
- metrics: the same three-class, AS-recall, component micro-F1, hit, exact, and
  scope definitions used by the V19 SFT evaluator.

Each graph node is one native V19 candidate. Candidate features average pinned
MiniLM embeddings of the candidate description and every referenced observable
event; the global node additionally collects any observable event not referenced
by a candidate, guaranteeing complete user-input coverage. Graph edges connect the global candidate, topology-linked node
candidates, edge candidates to their endpoint nodes, and tool candidates to
their owner nodes. No source placement or label field enters a feature.

The G-Safeguard-style row uses the official `MyGAT` encoder at commit
`890c99f1cbc864e9ff0c85859619a14f42bc9cab`. The BlindGuard-style row uses the
official `TAMModel` encoder at commit
`1889c20a326ba9ba9a6982744d473626e74f9986`. Both receive identical supervised
V19 verdict and localization heads because the original binary/agent-only heads
cannot express the V19 task.

## One-click validation

Run inside a scheduler job exposing exactly one GPU:

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull --ff-only origin main
bash SFT/auditor_agent_gnn_baselines_package/server_scripts/run_gnn_v19_marble.sh
```

The result archive is written outside the home directory:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v19_gnn_marble_validation.tar.gz
```

Do not run `run_gnn_v19_final_test_once.sh` until both validation rows and all
thresholds are frozen.
