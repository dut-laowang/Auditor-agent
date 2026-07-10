# V12 GNN-Distill MAS Auditor

This package keeps V12 as the main SFT baseline and adds a lightweight graph detector for GNN-style localization distillation.

Core idea:

```text
V12 sanitized MAS evidence -> graph detector -> component risk ranking -> SFT target
```

The final SFT input remains clean:

- no `ACI_` markers
- no attack ids
- no success markers
- no injected-instruction labels
- no verifier provenance
- no attack metadata in user-visible input

## What Changes From V12

V12 is not modified. This package generates a new distilled dataset at runtime:

```text
assistant.localization.candidate_ranking
```

The ranking is produced by a lightweight GNN-style graph detector trained on V12 graph candidates. It is used as assistant-side supervision only, not as user input.

## Scripts

- `build_gnn_graph_dataset.py`: convert V12 SFT rows into graph-detector data
- `train_light_gnn_baseline.py`: train/evaluate a pure PyTorch graph detector
- `build_distilled_sft.py`: build GNN-distilled SFT data from V12 + graph scores
- `train_qwen3_lora_sft.py`: Qwen3-8B LoRA SFT
- `eval_qwen3_fullschema.py`: SFT evaluation

## Expected Use

Use the one-shot command in `server_scripts/RUN_SERVER.md`.

Recommended comparison:

```text
V12 SFT baseline vs V12-GNN-Distill SFT
```

Main metrics:

- binary safe/unsafe F1
- safe F1 / unsafe F1
- three-class accuracy
- audit trace validity
- GNN run-level and component-level reports

This package does not claim guaranteed improvement before running. It provides the correct controlled experiment for testing whether graph-detector localization supervision improves V12.
