# V23 final experiments

This is the authoritative four-table experiment layer for the frozen
43,844-row V23 dataset. It does not alter dataset rows or schemas. Table 3 is a
labelled frozen V18/AgentForesight transfer reference; other core results come
from V23 outputs.

## One-command server run

From the uploaded `Auditor-agent` repository root:

```bash
export V23_DATA_DIR=/absolute/path/v23_final_aligned_combined
export V23_RUN=/absolute/path/v23_final_run
export V23_EXPERIMENT_RUN=/absolute/path/v23_final_experiment_run
bash SFT/auditor_agent_v23_final_experiments/server_scripts/run_v23_all_experiments.sh
```

The scheduler queries each GPU's physical memory and admits concurrent jobs by
an explicit reservation budget. On one 96GB H100, Qwen (34GB reservation) and
InternLM (34GB) start together while the scheduler retains roughly 15% for CUDA
peaks; ModernBERT (12GB) starts as soon as budget is available. Dependency-ready
CPU scoring overlaps automatically. For multiple cards,
set `V23_GPUS=0,1,...`. Override detection only when necessary with
`V23_GPU_MEMORY_GB`; tune the safety fraction with
`V23_GPU_MEMORY_UTILIZATION` (default `0.85`). Each stage resumes or skips only
after its completion marker exists. A failed task halts the suite, and required
`TBD` cells prevent the final completion marker.

The transfer artifact is written next to the result directory as
`v23_final_experiment_run_V23_FINAL_EXPERIMENT_RESULTS.tar.gz`, with a SHA-256
sidecar. Checkpoints are excluded.

## Authoritative scope

- Qwen3-8B and InternLM3-8B-Instruct V23 SFT
- ModernBERT zero-truncation training/evaluation
- fixed cascade, deterministic 15% rule router, and learned router
- held-out retraining for tree, message, and research folds
- official-source G-Safeguard, TAM, and XG-Guard adapters
- exactly four Markdown/LaTeX tables, with no imputed values

Files retaining `v22` in their names are frozen adapters or legacy references.
The entry point above does not call old orchestrators, closed-LLM scripts, or
V22 dataset runners.

Final markers are `V23_SFT_COMPLETE.json`, `INTERNLM3_SFT_COMPLETE.json`,
`MODERNBERT_COMPLETE.json`, `bounded_agent_common6167/V23_AGENT_COMPLETE.json`,
`components/COMPONENT_POLICIES_COMPLETE.json`, `SUPPLEMENT_SUITE_COMPLETE.json`,
and `V23_EXPERIMENTS_COMPLETE.json`.
