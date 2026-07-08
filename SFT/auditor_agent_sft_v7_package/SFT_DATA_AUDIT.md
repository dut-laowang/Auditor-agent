# SFT v7 Data Audit

## Task

V7 is an unknown-run MAS auditor dataset. The model receives graph, task, and run evidence, but it is not given attack surface, objective, or injection location in the user input.

The model must output:

- `decision.verdict`
- `decision.binary_label`
- `audit.inferred_attack`
- `audit.localization`
- `audit.evidence_refs`
- `audit.audit_trace`

## Label Standard

Labels are unchanged from V6/V5:

- marker-based offline judge
- `analysis_latest/success_runs.jsonl` when available
- generated marker scan for roots without analysis
- not strict final-output-only labeling

Markers are used only offline for labels and sanitization.

## Dataset

- total: 7362
- train: 5875
- test: 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

V6/V7 consistency:

- same run ids: true
- label mismatches: 0

## Input Sanitization

Forbidden in SFT-visible user messages:

- uppercase `ACI_...`
- lowercase `aci_...`
- `END_NEGOTIATION`
- `success_marker` / `success_markers`
- `attack_metadata`
- `attack_id`
- `marker_check`
- `[Injected ...]`
- `attack.surface`
- `attack.objective`
- `graph.attack_location`

Latest local validation:

- user `attack` field count: 0
- user `graph.attack_location` count: 0
- user `surface/objective` key count: 0
- synthetic suspicious-exposure event count: 0
- visible leak hits: 0
- bad evidence references: 0

## Evidence

V7 keeps V6 evidence granularity but removes explicit attack metadata from user input.

Observed event types include:

- `agent_act`
- `message_send`
- `tool_call`
- `tool_result`
- `run_summary`
- `final_outcome`
- `task_context`
