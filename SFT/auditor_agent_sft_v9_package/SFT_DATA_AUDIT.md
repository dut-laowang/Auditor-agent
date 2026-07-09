# SFT v9 Data Audit

V9 is an unknown-run MAS auditor dataset with graph-structured evidence. The user input contains graph, task, and sanitized run evidence only.

## No-Leak Policy

User-visible content must not expose:

- `attack_id`
- `attack_metadata`
- `success_marker`
- `ACI_...`
- `END_NEGOTIATION`
- `attack_location`
- `surface` / `objective` as attack labels
- `[Injected instruction]`
- `injected instruction`
- `benchmark addendum`
- verifier provenance terms

Current full-dataset check: 0 visible leak hits.

## Required Fields

Current full-dataset check:

- missing required fields: 0
- invalid evidence refs: 0
- non-empty trajectory not used: 0

## Evidence Format

V9 uses:

- `evidence.graph_evidence.global_events`
- `evidence.graph_evidence.node_events`
- `evidence.graph_evidence.edge_events`
- `evidence.graph_evidence.tool_events`
- `evidence.graph_evidence.final_outcome_events`
- `evidence.event_index`

Assistant `evidence_refs` and `audit_trace[*].evidence_ref` point to graph evidence ids such as `node.agent1.obs_0`, `edge.agent1->agent2.msg_0`, `tool.agent1.obs_0`, and `final.outcome_0`.

## Counts

- total: 7362
- train/test: 5875 / 1487
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697
- safe/unsafe: 3665 / 3697

## Evidence Coverage

- `has_run_trace`: 7354 / 7362
- `has_trajectory`: 3305 / 7362
- `trajectory_and_final_output`: 7354
- `final_output_only`: 1
- `context_only`: 7
- intermediate shard trajectories recovered: 1399
