# V18 Single-Agent / Trajectory Baselines

This package preregisters a fair comparison between trajectory-only auditors
and the V18 graph-grounded auditor.

## Methods

1. **AgentDoG 1.5 official**: frozen 8B checkpoint and official action-safety
   definition. Its score against V18 outcome labels is a transfer diagnostic,
   not a label-equivalent main result.
2. **AgentDoG 1.5 outcome-adapted**: the same frozen checkpoint with a fixed
   prompt asking whether the attack objective was actually achieved.
3. **V18-Flat**: Qwen3-8B LoRA trained on the same V18 split, targets, events,
   order, and hyperparameters as V18-Graph. It removes explicit topology,
   graph edges, typed candidates, and candidate/evidence grounding.
4. **V18-Graph/Ours**: the existing V18 observable model.

V18-Flat retains an unstructured list of legal component IDs only so that both
fine-tuned models can be evaluated with the same localization output space.

## Important label distinction

AgentDoG's native `unsafe` means that an unsafe action or decision occurred.
V18's `unsafe` means `attack_success`: the attack objective was achieved and
materially affected behavior or outcome. `attack_failed` is safe in V18.
Official and outcome-adapted AgentDoG results must therefore be reported
separately.

## Reproducibility

- identical V18 train/test run IDs and task-group split
- no graph-derived event selection
- no label-dependent formatting
- identical Qwen3-8B LoRA hyperparameters for Flat and Graph
- deterministic decoding
- restart-safe training and evaluation
- AgentDoG official repository inspected at commit
  `c8d803f267a43ec0e103a651265f50f1ff4456d5`
