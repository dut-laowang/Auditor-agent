# SFT v12 Data Audit

V12 is a candidate-based graph auditor dataset with short structured audit traces.

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

## Intended Change From V11

V12 keeps labels, sources, split, graph candidates, and localization targets consistent with V11.

It only reduces SFT noise:

- replaces `evidence.global_events` / `clean_reference_events` with `run_evidence.observed` / `final_output` / `reference`
- removes assistant top-level `evidence_refs`
- replaces verbose audit-trace claims with two short structured steps

## Candidate Localization

Each graph candidate has a stable ID:

- `G::run`
- `N::agent`
- `E::source->target`
- `T::agent`

Assistant `localization.component_ids` selects from these IDs.

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
