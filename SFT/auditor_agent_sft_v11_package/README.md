# Candidate-based MAS Auditor SFT v11

V11 keeps the V10 data sources, marker-based labels, train/test split, and no-leak policy unchanged.

The intended change is the graph localization formulation:

```text
V10: model freely generates nodes / edges / tools
V11: input provides sanitized graph candidates, model selects component_ids
```

This makes localization closer to graph node/edge classification while still using Qwen instruction SFT.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

## User Input

The user-visible input contains:

- task and graph topology
- sanitized global/final/clean evidence
- `graph_candidates`: global, node, edge, and tool candidates with event references
- `audit_request.mode = unknown_run_audit`

It does not expose attack ids, success markers, injected-instruction labels, verifier provenance, or attack metadata.

## Assistant Output

```json
{
  "decision": {
    "verdict": "clean_safe | attack_failed | attack_success",
    "binary_label": "safe | unsafe"
  },
  "attack": {
    "present": true,
    "surface": "none | input | memory | message | profile | tool | dual",
    "objective": "none | disclosure | disruption | hijacking"
  },
  "localization": {
    "scope": "none | global | node | edge | tool | mixed",
    "component_ids": ["G::run", "N::agent1", "E::agent1->agent2", "T::agent1"]
  },
  "evidence_refs": [],
  "audit_trace": []
}
```

## Quality Gates

- schema: `Graph-grounded-Candidate-SFT/v11`
- visible leak hits: 0
- all non-mixed localization targets map to candidate IDs
- average candidates: about 13.4 per sample
- max candidates: 87 after dense-graph pruning
- manual random quality sample: `sft_dataset_graph_grounded_v11/manual_quality_sample_50_v11.json`

## Files

- `scripts/convert_v10_to_v11.py`: reproduces the V10-to-V11 candidate conversion
- `sft_dataset_graph_grounded_v11/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts
