# Decision-first Graph-grounded MAS Auditor SFT v4

This package contains a v4 SFT dataset and training/evaluation workflow for MAS safety auditing.

## What Changed From v3

- v4 uses a decision-first assistant target:
  - `decision`: verdict, binary label, risk, action.
  - `audit`: localization, evidence references, short audit trace.
- The final verdict is emitted before explanation to reduce classification/explanation interference.
- `audit_trace.claim` is short and does not copy long evidence text.
- Evidence is cited by reference ids such as `obs_0`, `clean_0`, and `graph.attack_location`.
- The no-leak policy from v3 is retained.

## Contents

- `scripts/build_graph_grounded_sft.py`
  - Converts MARBLE benchmark artifacts to v4 SFT JSONL.
- `sft_dataset_graph_grounded_v4/`
  - `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`.
- `sft_samples_graph_grounded_v4/`
  - Six readable Minecraft/Bargaining examples.
- `server_scripts/`
  - `train_qwen3_lora_sft.py`: Qwen3-8B LoRA SFT.
  - `eval_qwen3_fullschema.py`: Base/SFT evaluation with verdict metrics and audit-trace quality.
  - `compare_eval_metrics.py`: compare two `metrics.json` files.
  - `RUN_SERVER.md`: concise server commands.

## Data

Current generated dataset:

- total: 1460
- train/test: 1162 / 298
- labels: safe 1074, unsafe 386
- verdicts: clean_safe 48, attack_failed 1026, attack_success 386

v4-main uses the original marker-based offline judge label from `analysis_latest/success_runs.jsonl`. This keeps labels aligned with v2 while changing only the SFT output format to decision-first audit trace.

Visible SFT messages do not contain uppercase success markers, `success_marker(s)`, `attack_metadata`, `attack_id`, `[Injected ...]` labels, `END_NEGOTIATION`, `offline verifier`, or `attack-success index`.

## Regenerate

```bash
python scripts/build_graph_grounded_sft.py \
  --zip benchmark384_multiscenario_runs_1.zip \
  --output-dir sft_dataset_graph_grounded_v4 \
  --test-ratio 0.2 \
  --seed 42
```
