# Run V19 Quality Fix

Use the same Qwen3-8B LoRA environment as V18. This one command restores and
verifies all three datasets, trains three independent adapters, runs clean and
counterfactual validation, consumes each sealed test exactly once, aggregates
the three result rows, hashes the result tree, and creates one transfer archive.

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor
REPO=$BASE/Auditor-agent
cd "$REPO"
git pull --ff-only origin main
GPU=0 BASE="$BASE" REPO="$REPO" \
  bash SFT/auditor_agent_sft_v19_qualityfix_package/server_scripts/run_v19_all_experiments_once.sh
```

All outputs are placed under one timestamped folder:

```text
$BASE/v19_three_track_results_<UTC timestamp>/
```

The adjacent archive is ready to transfer back:

```text
$BASE/v19_three_track_results_<UTC timestamp>.tar.gz
```

The folder contains each track's adapter, training logs, lexical proxy,
counterfactual data/results, final predictions/metrics, sealed-test record, the
three-track average, and a root `SHA256SUMS`.
