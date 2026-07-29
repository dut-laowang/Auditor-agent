# MAS Auditor SFT V17 Semantic-Preserving

V17 rebuilds the completed `weekend_fresh_single1_10_dual1_5` benchmark with
semantic-preserving redaction. It uses the same source labels and task-group
split policy as V16.1, without mixing any V12 training data.

## Data policy

- Source `final_label` is immutable.
- `success` maps to `attack_success`; `failure` maps to `attack_failed`.
- `not_exposed`, `ambiguous`, and `invalid` remain in the exclusion audit.
- Eligible clean and private-control runs map to `clean_safe`.
- Raw markers, private values, attack IDs, benchmark injection tags, and gold
  labels never appear in SFT-visible input.
- Each sensitive entity receives a deterministic per-run pseudonym. The same
  entity remains linkable within one run but cannot be matched across runs.
- Ordinary instruction, communication, tool, and outcome semantics are retained.
- Marker-derived judge events and clean-reference label proxies remain excluded.
- No `(scenario, sample_id)` group occurs in both train and test.
- Localization remains a projection of source attack placement, not a newly
  adjudicated causal label.

## Output

The assistant generates:

```text
decision + attack + localization + audit_trace
```

Evaluation reports three-class and binary performance, prediction distributions,
confusion matrices, component localization, per-scope localization, and evidence
reference validity.

## Rebuild

```powershell
python scripts/build_v17_semantic_dataset.py `
  --source-root D:\FIRST_COPILOT\plan_e\_v16_source_extract\weekend_fresh_single1_10_dual1_5 `
  --output-dir sft_dataset_graph_grounded_v17_semantic `
  --v15-builder ..\auditor_agent_sft_v15_hq_current_package\scripts\build_v15_hq_current_dataset.py
```

See `server_scripts/RUN_SERVER.md` for the server workflow.
