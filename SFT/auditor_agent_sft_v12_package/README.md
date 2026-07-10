# Candidate-based MAS Auditor SFT v12

V12 keeps the V10/V11 sources, marker-based labels, train/test split, and no-leak policy unchanged.

The intended change from V11 is format cleanup only:

```text
V11: evidence.global_events / clean_reference_events + verbose audit_trace
V12: run_evidence.observed / final_output / reference + short structured audit_trace
```

The graph localization remains candidate-based:

```text
graph_candidates -> localization.component_ids
```

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

- schema: `Graph-grounded-Candidate-SFT/v12`
- visible leak hits: 0
- old user `evidence` field removed
- assistant top-level `evidence_refs` removed
- trace refs valid: 100%
- average candidates: about 13.4 per sample
- max candidates: 87 after dense-graph pruning
- manual random quality sample: `sft_dataset_graph_grounded_v12/manual_quality_sample_50_v12.json`

## Files

- `scripts/convert_v10_to_v12.py`: reproduces the V10-to-V12 format cleanup
- `sft_dataset_graph_grounded_v12/`: `all.jsonl`, `train.jsonl`, `test.jsonl`, `stats.json`
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts
