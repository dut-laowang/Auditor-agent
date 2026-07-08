# Decision-first Graph-grounded MAS Auditor SFT v6

V6 is a minimal upgrade over V5. It keeps the same task, label policy, split logic, and decision-first output format, but improves evidence granularity and audit-trace specificity.

## What Changed From V5

- Same run ids as V5.
- Same marker-based labels as V5/V4-main.
- Same assistant top-level format: `decision` then `audit`.
- New input `evidence.coverage`.
- More fine-grained observed events:
  - `attack_exposure`
  - `agent_act`
  - `message_send`
  - `tool_call`
  - `tool_result`
  - `run_summary`
  - `final_outcome`
- More specific audit trace steps:
  - `localize_exposure`
  - `inspect_agent_behavior`
  - `inspect_final_outcome`
  - `compare_with_clean`
  - `decide`

Markers are still used only for offline labeling and localization. They are not exposed in SFT-visible messages.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- binary labels: safe 3665, unsafe 3697
- verdicts: clean_safe 192, attack_failed 3473, attack_success 3697
- scenarios: bargaining 1847, coding 1846, minecraft 1848, research 1821
- topologies: chain 1830, graph 1848, star 1837, tree 1847

V5/V6 consistency check:

- same run ids: true
- label mismatches vs V5: 0

## Evidence Validation

- schema: `Graph-grounded-Evidence-SFT/v6`
- visible leak hits: 0
- bad structure: 0
- empty observed evidence: 0
- bad evidence references: 0
- average observed events: 4.691

Granularity:

- final_output_only: 5449
- trajectory_and_final_output: 1906
- context_only: 7

## Files

- `scripts/build_graph_grounded_sft.py`: V6 converter.
- `sft_dataset_graph_grounded_v6/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`.
- `sft_samples_graph_grounded_v6/`: six readable examples.
- `server_scripts/`: Qwen3-8B LoRA training, evaluation, and comparison scripts.

Server commands are in `server_scripts/RUN_SERVER.md`.
