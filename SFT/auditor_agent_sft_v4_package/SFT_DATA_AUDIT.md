# SFT v4 Data Audit

Generated from `benchmark384_multiscenario_runs_1.zip`.

## Dataset

- total: 1460
- train: 1162
- test: 298
- safe: 1164
- unsafe: 296
- clean_safe: 48
- attack_failed: 1116
- attack_success: 296

The upstream offline success index has 386 successes. v4 keeps 296 strict observable successes after removing cases where the signal appears only in task/config/prompt text or injected blocks.

## Leak Policy

The converter validates only SFT-visible `messages`, not metadata. Training removes metadata before tokenization.

Forbidden in visible messages:

- uppercase `ACI_...`
- `END_NEGOTIATION`
- `success_marker` / `success_markers`
- `attack_metadata`
- `attack_id`
- `marker_check`
- `[Injected ...]`
- `offline verifier`
- `attack-success index`
- `labeled as attack-success`

Latest local validation:

- visible leak hits: 0
- schema: `Graph-grounded-Evidence-SFT/v4`
- assistant structure: `decision` first, `audit` second
- audit-trace bad evidence references: 0
- audit-trace evidence-ref validity: 100%

## Output Structure

Each assistant answer contains:

- `decision`
  - `verdict`
  - `binary_label`
  - `risk_level`
  - `safe`
  - `action`
- `audit`
  - `localization`
  - `evidence_refs`
  - `audit_trace`

`audit_trace` uses short claims and cites references from the user input, for example:

- `task.goal`
- `graph.attack_location`
- `attack.objective`
- `attack.surface`
- `obs_0`, `obs_1`, ...
- `clean_0`, `clean_1`, ...

v4 is designed to reduce classification/explanation interference by emitting the decision before the audit trace.
