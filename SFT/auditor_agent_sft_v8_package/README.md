# Unknown-run Graph-grounded MAS Auditor SFT v8

V8 is a data-quality-first candidate for unknown-run MAS safety auditing. The model input contains only graph, task, and sanitized run logs/evidence; the target output predicts attack success, inferred attack type, localization, and cited audit trace.

Compared with V7, V8 keeps the same marker-based labels and no-leak user input policy, but improves evidence extraction:

- reads `final/merged/trajectories` when available;
- falls back to `intermediate/shards/*/trajectories` when merged trajectories are empty;
- extracts non-label agent/message evidence from `marble_results`;
- removes visible attack ids, markers, verifier text, and injection labels from user input.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

## Evidence Quality

- user leak hits: 0
- `has_run_trace`: 7354 / 7362
- `has_trajectory`: 3305 / 7362
- `trajectory_and_final_output`: 7354
- `final_output_only`: 1
- `context_only`: 7
- intermediate trajectories recovered: 1399

The remaining `context_only`/`final_output_only` cases are listed in `sft_dataset_graph_grounded_v8/trajectory_quality_report.json`.

## Files

- `scripts/build_graph_grounded_sft.py`: V8 converter.
- `sft_dataset_graph_grounded_v8/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`, `trajectory_quality_report.json`, `manual_quality_sample_60.json`.
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts.
