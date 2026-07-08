# Unknown-run Graph-grounded MAS Auditor SFT v7

V7 is a minimal deployment-style adaptation of V6.

V6 answers: given a known attack condition, did the attack succeed?

V7 answers: given a MAS graph, task, and run evidence with unknown safety status, infer whether an attack is present/successful and produce an audit report.

## What Changed From V6

- Same run ids, labels, train/test split, and dataset size as V6.
- User input no longer exposes:
  - `attack`
  - `attack.present`
  - `attack.surface`
  - `attack.objective`
  - `graph.attack_location`
- User input still contains graph, task, run evidence, clean contrast when available, and sanitized suspicious-context evidence.
- Assistant output adds `audit.inferred_attack`.
- Labels remain marker-based and identical to V6.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

V6/V7 consistency:

- same run ids: true
- label mismatches: 0

## Validation

- schema: `Graph-grounded-Evidence-SFT/v7`
- user `attack` fields: 0
- user `graph.attack_location`: 0
- user `surface/objective` keys: 0
- synthetic suspicious-exposure hints: 0
- visible leak hits: 0
- assistant has `inferred_attack`: 7362 / 7362
- bad evidence references: 0

## Files

- `scripts/build_graph_grounded_sft.py`: V7 converter.
- `sft_dataset_graph_grounded_v7/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`.
- `sft_samples_graph_grounded_v7/`: six readable examples.
- `server_scripts/`: Qwen3-8B LoRA training, full evaluation, 200-sample quick evaluation, and comparison scripts.

Server commands are in `server_scripts/RUN_SERVER.md`.
