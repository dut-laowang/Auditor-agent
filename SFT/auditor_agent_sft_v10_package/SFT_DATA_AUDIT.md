# SFT v10 Data Audit

V10 is an unknown-run MAS auditor dataset with the same graph-structured user evidence as V9 and a slimmer assistant output schema.

## No-Leak Policy

User-visible content must not expose:

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

Current full-dataset check: 0 visible leak hits.

## Single Intended Change From V9

V10 changes only the assistant target format. It removes duplicated or non-core fields and keeps:

- `decision`
- `attack`
- `localization`
- `evidence_refs`
- `audit_trace`

The user input graph/evidence, labels, split, and no-leak policy are unchanged.

## Evidence Format

V10 keeps the V9 input evidence format:

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
