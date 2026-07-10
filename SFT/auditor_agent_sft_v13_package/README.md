# Candidate-based MAS Auditor SFT v13

V13 keeps the V12 data sources, marker-based labels, train/test split, `run_evidence`, `graph_candidates`, and no-leak policy unchanged.

The intended change from V12 is output alignment:

```text
V12: decision + attack + localization.scope/component_ids
V13: decision contains all classification labels; localization only contains component_ids
```

This makes verdict, binary label, attack presence, surface, objective, and scope a unified classification target.

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
- `run_evidence.observed`
- `run_evidence.final_output`
- optional `run_evidence.reference.clean`
- `graph_candidates`: global, node, edge, and tool candidates with evidence references
- `audit_request.mode = unknown_run_audit`

It does not expose attack ids, success markers, injected-instruction labels, verifier provenance, or attack metadata.

## Assistant Output

```json
{
  "decision": {
    "verdict": "clean_safe | attack_failed | attack_success",
    "binary_label": "safe | unsafe",
    "attack_present": true,
    "surface": "none | input | memory | message | profile | tool | dual",
    "objective": "none | disclosure | disruption | hijacking",
    "scope": "none | global | node | edge | tool | mixed"
  },
  "localization": {
    "component_ids": ["G::run", "N::agent1", "E::agent1->agent2", "T::agent1"]
  },
  "audit_trace": [
    {
      "step": "localize_component",
      "component_refs": ["E::agent1->agent2"],
      "evidence_refs": ["obs_3"]
    },
    {
      "step": "verify_outcome_effect",
      "component_refs": ["E::agent1->agent2"],
      "evidence_refs": ["out_0", "ref_0"]
    }
  ]
}
```

## Quality Gates

- schema: `Graph-grounded-Candidate-SFT/v13`
- visible leak hits: 0
- old assistant `attack` field removed
- old `localization.scope` removed
- all six decision keys present in every sample
- trace refs valid: 100%
- average candidates: about 13.4 per sample
- max candidates: 87 after dense-graph pruning
- manual random quality sample: `sft_dataset_graph_grounded_v13/manual_quality_sample_50_v13.json`

## Files

- `scripts/convert_v12_to_v13.py`: reproduces the V12-to-V13 output alignment
- `sft_dataset_graph_grounded_v13/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts
