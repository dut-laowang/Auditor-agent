# SFT v6 Data Audit

## Label Standard

V6 uses the same marker-based label standard as V5/V4-main:

- Existing `analysis_latest/success_runs.jsonl` is used when available.
- Missing analysis roots are labeled by generated marker scan over run outputs.
- Marker strings are used only for offline labeling and sanitization.
- This is not the stricter final-output-only label variant.

V5/V6 label check:

- V5 rows: 7362
- V6 rows: 7362
- same run ids: true
- label mismatches: 0

## Dataset

- total: 7362
- train: 5875
- test: 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

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
- evidence references: 52822
- bad evidence references: 0

## Evidence

V6 adds `evidence.coverage` and finer event types while keeping the V5 decision-first assistant format.

Observed event stats:

- average observed events: 4.691
- min observed events: 1
- attack_exposure: 7170
- run_summary: 12412
- final_outcome: 7355
- trajectory-backed samples: 1906
- final-output-only samples: 5449
- context-only samples: 7

## Assistant Format

Each assistant answer remains:

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

The main V6 change is that `audit_trace` now cites finer evidence types such as `attack_exposure`, `run_summary`, and `final_outcome`.
