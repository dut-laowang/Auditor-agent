# SFT v11 Data Audit

V11 is an unknown-run MAS auditor dataset with candidate-based graph localization.

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

## Intended Change From V10

V11 keeps V10 labels, sources, and split. It changes localization from open generation to candidate selection:

```text
V10 output: localization.nodes / localization.edges / localization.tools
V11 input:  graph_candidates
V11 output: localization.component_ids
```

This is designed to reduce graph-text noise and make SFT closer to node/edge/tool classification.

## Evidence Format

User-visible input contains:

- `evidence.global_events`
- `evidence.clean_reference_events`
- `graph_candidates`

Each graph candidate has a stable ID:

- `G::run`
- `N::agent`
- `E::source->target`
- `T::agent` or `T::agent::tool`

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
- non-mixed localization candidate coverage: 100%
