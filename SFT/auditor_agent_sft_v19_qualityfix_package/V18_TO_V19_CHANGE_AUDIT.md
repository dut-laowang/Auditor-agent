# V18 to V19 minimal-change audit

## Unchanged research/model framework

- base model: Qwen3-8B;
- adaptation: LoRA with `r=16`, alpha 32, dropout 0.05;
- target modules: q/k/v/o, gate/up/down projections;
- assistant-only SFT supervision and chat-template preprocessing;
- 2 epochs, learning rate `2e-4`, effective batch configuration 2 x 8;
- observable user-input policy and per-run secret redaction;
- three-way verdict: clean-safe / attack-failed / attack-success;
- attack surface/objective, graph localization, and JSON audit trace outputs;
- deterministic greedy full-schema evaluation.

No graph neural network, policy encoder, new backbone, new loss, reward model,
or post-hoc classifier was introduced.

## Targeted quality changes only

1. dual placements project all resolvable source components;
2. evidence references use localized candidate events and observable outcomes;
3. train/validation/test replace the V18 train/test-only protocol;
4. training uses validation, never final test;
5. fixed star replaces legacy star; AutoGen is exposed as a separate track;
6. deterministic seeds, hashes, adapter manifest, and logs are recorded;
7. validation-only shortcut and counterfactual experiments are required;
8. surface/objective metrics are reported in addition to V18 metrics.

The builder has more code because it supports multiple source layouts, three
splits, integrity gates, and dual/evidence projection. This is data engineering
and evaluation hardening, not a change to the learned auditor architecture.

## Deferred item

Human review remains pending. No manual-QC pass claim is allowed.
