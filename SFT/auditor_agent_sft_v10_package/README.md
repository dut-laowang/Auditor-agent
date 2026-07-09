# Graph-structured MAS Auditor SFT v10

V10 keeps the V9 input evidence, labels, train/test split, no-leak policy, and graph-grounded task setting unchanged.

The only intended change is the assistant output schema: V10 removes duplicated localization fields and keeps one slim graph localization target.

## Dataset

- total: 7362
- train/test: 5875 / 1487
- safe: 3665
- unsafe: 3697
- clean_safe: 192
- attack_failed: 3473
- attack_success: 3697

## Output Schema

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
    "nodes": [],
    "edges": [],
    "tools": []
  },
  "evidence_refs": [],
  "audit_trace": []
}
```

Removed from V9 output:

- duplicated `audit.inferred_attack.location`
- duplicated `audit.localization.graph_target`
- duplicated `affected_nodes` / `affected_edges`
- `risk_level`, `safe`, `action`
- `source_agent`, `target_agent`, `selector`

## Quality Gates

- schema: `Graph-grounded-Evidence-SFT/v10`
- train/test split unchanged from V9
- visible leak hits: 0
- assistant key shape: exactly `decision`, `attack`, `localization`, `evidence_refs`, `audit_trace`

## Files

- `scripts/convert_v9_to_v10.py`: reproduces the V9-to-V10 assistant-schema slimming step
- `sft_dataset_graph_grounded_v10/`: `all.jsonl`, `train.jsonl`, `test.jsonl`
- `server_scripts/`: Qwen3-8B LoRA training/evaluation scripts
- `server_scripts/make_subset_by_ids.py`: builds V10 eval subset from an existing run-id list
