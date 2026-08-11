# V19 MARBLE graph baselines

This package evaluates three MAS graph defenses on the identical frozen V19
MARBLE split and native `G::` / `N::` / `E::` / `T::` localization space:

1. G-Safeguard (supervised, V19-adapted)
2. BlindGuard (normal-only self-supervised, V19-adapted)
3. XG-Guard (normal-only bi-level graph anomaly detection, V19-adapted)

The validation run consumes train (4,565 rows) and validation (1,791 rows).
The 1,491-row test remains sealed and has a separate one-time runner. Dataset
SHA-256 values, official source commits, checkpoints, calibration thresholds,
and predictions are recorded in the output contracts.

## One-click validation run

Run this inside a scheduler allocation exposing one GPU. Do not prefix the
command with `GPU=0` or `CUDA_VISIBLE_DEVICES=...`; the scheduler-selected GPU
is used directly.

```bash
cd /gs/bs/tgh-26IAW/hongbo/project_4_coauthor/Auditor-agent
git pull --ff-only origin main
conda activate mou
bash SFT/auditor_agent_gnn_baselines_package/server_scripts/run_gnn_v19_marble.sh
```

Models and caches stay below
`/gs/bs/tgh-26IAW/hongbo/project_4_coauthor`, never under the home directory.
The transferable result archive is:

```text
/gs/bs/tgh-26IAW/hongbo/project_4_coauthor/v19_gnn_marble_validation.tar.gz
```

The publication-ready rows are in:

```text
v19_gnn_marble_validation/main_table_rows.tsv
```

See `V19_GNN_BASELINES.md` for the exact fairness and adaptation contract.
