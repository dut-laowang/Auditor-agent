# Graph-structured MAS Auditor SFT v9

V9 is a local candidate for unknown-run MAS safety auditing.

Compared with V8, V9 keeps the same labels, split, and no-leak policy, but changes the user-side evidence format from a flat event list to graph-structured evidence:

- `global_events`
- `node_events`
- `edge_events`
- `tool_events`
- `final_outcome_events`

The model input contains only graph, task, and sanitized MAS logs/evidence. It does not expose attack ids, markers, success markers, attack metadata, attack locations, attack surfaces, or attack objectives.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

## Quality Gates

- schema: `Graph-grounded-Evidence-SFT/v9`
- missing required fields: 0
- visible leak hits: 0
- invalid evidence refs: 0
- non-empty trajectory not used: 0
- manual 50-sample audit problems: 0
- localization scope mismatches: 0

Localization is graph-normalized:

- `input` -> `graph_target.scope = global`
- `memory/profile` -> `graph_target.scope = node`
- `message` -> `graph_target.scope = edge`
- `tool` -> `graph_target.scope = tool`
- `dual` -> `graph_target.scope = mixed`
- `none` -> `graph_target.scope = none`

## Evidence Coverage

- `has_run_trace`: 7354 / 7362
- `has_trajectory`: 3305 / 7362
- `trajectory_and_final_output`: 7354
- `final_output_only`: 1
- `context_only`: 7
- intermediate trajectories recovered: 1399

## Files

- `scripts/build_graph_grounded_sft.py`: V9 converter.
- `sft_dataset_graph_grounded_v9/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`, `trajectory_quality_report.json`, `manual_quality_sample_50_current.json`.
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts.
