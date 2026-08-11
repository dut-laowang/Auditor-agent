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
by a candidate, guaranteeing complete semantic user-input coverage. Long pieces
are tokenized without truncation, split into encoder-sized windows, fully encoded,
and token-count-weighted before candidate aggregation. Graph edges connect the global candidate, topology-linked node
candidates, edge candidates to their endpoint nodes, and tool candidates to
their owner nodes. No source placement or label field enters a feature.

The package now contains three deliberately different learning contracts:

- **G-Safeguard (V19-adapted)** uses official `MyGAT` at commit
  `890c99f1cbc864e9ff0c85859619a14f42bc9cab` with supervised V19 verdict,
  scope, and component heads.
- **BlindGuard (V19-adapted)** uses official `GATSCL` at commit
  `1889c20a326ba9ba9a6982744d473626e74f9986`. It trains only on V19
  `clean_safe` rows and preserves directional synthetic corruption,
  self-supervised contrastive loss, and similarity-based anomaly scoring.
- **XG-Guard (V19-adapted)** follows official commit
  `86e1121512f76800f80d4687e492c7f99f049929`, preserving its sentence/token
  bi-level GCN, discussion-theme contexts, negative context permutation, and
  score fusion. Long V19 records are chunked without truncation. Chunk means
  plus token counts are retained; this is an exact sufficient statistic for
  the released model's mean token anomaly score.

BlindGuard and XG-Guard never optimize or calibrate against V19 attack labels.
Two V19 three-way cutoffs (normal-score quantiles 0.95/0.99) and the component
cutoff (0.90) are frozen solely from the `clean_safe` training-score
distribution. Validation labels are used only for metrics. Their scope is derived
from selected G/N/E/T candidates. This projection is an adaptation, not a claim
that either original method natively predicts the V19 schema. TAM is not part
of this package or table.

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

`main_table_rows.tsv` contains the three publication-table-ready Clean rows. Its
accuracy delta is computed against the frozen V19 SFT (Qwen3-8B) accuracy of
75.60%, matching the existing main table.

Do not run `run_gnn_v19_final_test_once.sh` until both validation rows and all
thresholds are frozen.
