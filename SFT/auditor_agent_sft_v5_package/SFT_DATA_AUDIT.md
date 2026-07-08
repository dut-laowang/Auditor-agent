# SFT v5 Data Audit

## Dataset

- total: 7362
- train: 5875
- test: 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

Sources:

- canonical: 6601
- dual: 601
- noncanonical: 160

The converter preserves `source_zip`, `source_root`, `source_type`, `sample_id`, `scenario`, `topology`, `surface`, `objective`, and `placement` in metadata for split analysis. Training code removes metadata before tokenization.

## Label Standard

Labels follow the current marker-based judge standard:

- Existing `analysis_latest/success_runs.jsonl` is used when available.
- Missing analysis roots are labeled by generated marker scan over run outputs.
- Marker strings are used only for offline labeling and sanitization.

This is not the stricter final-output-only label variant.

## Visible Input Policy

Forbidden in SFT-visible `messages`:

- uppercase `ACI_...`
- lowercase `aci_...`
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
- bad message structure: 0
- empty observed evidence: 0
- evidence references: 34515
- bad evidence references: 0
- schema: `Graph-grounded-Evidence-SFT/v5`

## Assistant Format

Each assistant answer is JSON:

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

`audit_trace` uses short claims and cites input references such as `obs_0`, `clean_0`, `graph.attack_location`, `attack.objective`, and `attack.surface`.
