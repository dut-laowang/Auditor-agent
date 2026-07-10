# SFT v13 Data Audit

V13 is a candidate-based graph auditor dataset with evaluation-aligned decision labels.

## No-Leak Policy

User-visible content must not expose:

- `attack_id`
- `attack_metadata`
- `success_marker`
- `success_markers`
- `ACI_...`
- `[Injected instruction]`
- verifier provenance terms
- the gold verdict or gold localization target as explicit user fields

Current full-dataset check: 0 visible leak hits for the protected patterns above.

## Intended Change From V12

V13 keeps labels, sources, split, `run_evidence`, graph candidates, localization targets, and audit traces consistent with V12.

It only moves classification labels into `decision`:

- `attack.present` -> `decision.attack_present`
- `attack.surface` -> `decision.surface`
- `attack.objective` -> `decision.objective`
- `localization.scope` -> `decision.scope`
- `localization` keeps only `component_ids`

## Counts

- total: 7362
- train/test: 5875 / 1487
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697
- safe/unsafe: 3665 / 3697
- average candidates: about 13.4
- max candidates: 87
- trace invalid refs: 0
- assistant old attack field rows: 0
- assistant old localization.scope rows: 0
