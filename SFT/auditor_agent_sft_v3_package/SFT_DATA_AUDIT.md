# SFT v3 Data Audit

Generated from `benchmark384_multiscenario_runs_1.zip`.

## Dataset

- total: 1460
- train: 1162
- test: 298
- safe: 1163
- unsafe: 297
- clean_safe: 48
- attack_failed: 1115
- attack_success: 297

The upstream offline success index has 386 successes. v3 keeps 297 strict observable successes after removing cases where the signal appears only in task/config/prompt text or injected blocks.

## Leak Policy

The converter validates only SFT-visible `messages`, not metadata. Training removes metadata before tokenization.

Forbidden in visible messages:

- uppercase `ACI_...`
- `END_NEGOTIATION`
- `success_marker` / `success_markers`
- `attack_metadata`
- `attack_id`
- `marker_check`
- `[Injected instruction]`
- `offline verifier`
- `attack-success index`
- `labeled as attack-success`

Latest local validation:

- visible leak hits: 0
- schema: `Graph-grounded-Evidence-SFT/v3`
- audit-trace bad evidence references: 0
- audit-trace evidence-ref validity: 100%

## Evidence Trace

Each assistant answer contains:

- `verdict`
- `label`
- `risk_level`
- `localization`
- `audit_trace`
- `evidence_refs`
- `final_answer`

`audit_trace` cites references from the user input, for example:

- `task.goal`
- `graph.attack_location`
- `attack.objective`
- `attack.surface`
- `obs_0`, `obs_1`, ...
- `clean_0`, `clean_1`, ...

This is an automated expert-style audit trace: stronger than v2 templates, but still derived from available artifacts rather than manual expert annotation.
