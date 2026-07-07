# Graph-grounded MAS Auditor SFT v3

This package upgrades v2 into a graph-grounded, evidence-cited audit-trace SFT dataset and training/evaluation workflow.

## What Changed From v2

- The v2 `verdict + evidence + reason` target is replaced by structured `audit_trace`.
- Each audit step cites input evidence references such as `obs_0`, `clean_0`, `graph.attack_location`, and `task.goal`.
- User-visible `run_id` is anonymized as `sample_uid`; real run ids stay only in metadata and are removed before training.
- The converter enforces a stricter no-leak policy for markers, attack ids, injected-instruction tags, and verifier provenance.
- Evaluation now reports both F1 metrics and audit-trace quality metrics.

## Contents

- `scripts/build_graph_grounded_sft.py`
  - Converts MARBLE benchmark artifacts to v3 SFT JSONL.
- `sft_dataset_graph_grounded_v3/`
  - `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`.
- `sft_samples_graph_grounded_v3/`
  - Six readable Minecraft/Bargaining examples.
- `server_scripts/`
  - `train_qwen3_lora_sft.py`: Qwen3-8B LoRA SFT. Supports `--init-adapter` for continuing from v2 LoRA.
  - `eval_qwen3_fullschema.py`: Base/SFT evaluation with verdict metrics and audit-trace quality.
  - `RUN_SERVER.md`: concise server commands.

## Data

Current generated dataset:

- total: 1460
- train/test: 1162 / 298
- labels: safe 1163, unsafe 297
- verdicts: clean_safe 48, attack_failed 1115, attack_success 297

The original offline success index contains 386 successes. v3 uses a stricter observable-label policy and keeps 297 successes: a success signal must appear in non-injected trajectory events or observable MARBLE result fields. Task/config/prompt text and injected blocks are ignored.

Visible SFT messages do not contain uppercase success markers, `success_marker(s)`, `attack_metadata`, `attack_id`, `[Injected instruction]`, `END_NEGOTIATION`, `offline verifier`, or `attack-success index`.

## Regenerate

```bash
python scripts/build_graph_grounded_sft.py \
  --zip benchmark384_multiscenario_runs_1.zip \
  --output-dir sft_dataset_graph_grounded_v3 \
  --test-ratio 0.2 \
  --seed 42
```

The converter expects the zip to contain merged configs, trajectories, MARBLE results, run manifest, and success runs from the benchmark output.
