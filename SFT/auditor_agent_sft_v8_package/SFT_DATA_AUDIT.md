# SFT v8 Data Audit

V8 is an unknown-run MAS auditor dataset. The user input contains graph, task, and sanitized run evidence, but does not expose attack ids, markers, success markers, attack metadata, or attack locations.

## Core Counts

- total: 7362
- train/test: 5875 / 1487
- safe/unsafe: 3665 / 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

## Evidence Coverage

- `has_run_trace`: 7354 / 7362
- `has_trajectory`: 3305 / 7362
- `trajectory_and_final_output`: 7354
- `final_output_only`: 1
- `context_only`: 7
- intermediate shard trajectories recovered: 1399

## No-Leak Check

User-visible hits are zero for:

- `attack_id`
- `attack_metadata`
- `success_marker`
- `ACI_...`
- `END_NEGOTIATION`
- `attack_location`
- `[Injected instruction]`
- `injected instruction`
- `benchmark addendum`
- verifier provenance terms

## Quality Caveat

Some source trajectory files are genuinely empty. V8 compensates by extracting non-label agent/message evidence from `marble_results`, but `trajectory_quality_report.json` still records every run whose raw trajectory file has no usable events.
