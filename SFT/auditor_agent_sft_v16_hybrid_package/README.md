# MAS Auditor SFT V16-Hybrid

V16 keeps the V12 graph-grounded SFT schema and converts the completed
`weekend_fresh_single1_10_dual1_5` benchmark into a uniform training corpus.
It does not relabel the source benchmark.

## Frozen policies

- Source `final_label` is immutable.
- `success` is represented as `attack_success`.
- `failure` is represented as `attack_failed`.
- `not_exposed`, `ambiguous`, and `invalid` are retained in
  `excluded_source_labels.jsonl`; they are not forced into the three-class target.
- Completed clean runs are `clean_safe`.
- Completed private controls are `clean_safe` only when the supplied
  private-control signal reports no natural marker leakage.
- Localization is the source attack-placement projection into the V12
  global/node/edge/tool candidate space. It is not claimed to be a new causal label.
- The split is task-grouped: no `(scenario, sample_id)` occurs in both train and test.

## Output schema

The SFT-visible input contains only sanitized task, graph, candidates, trajectory
evidence, and final-output evidence. It excludes attack IDs, payload metadata,
success markers, raw private values, injected-instruction labels, and gold labels.

The assistant target remains:

```text
decision + attack + localization + audit_trace
```

## Rebuild

```powershell
python scripts/build_v16_hybrid_dataset.py `
  --source-root D:\FIRST_COPILOT\plan_e\_v16_source_extract\weekend_fresh_single1_10_dual1_5 `
  --output-dir sft_dataset_graph_grounded_v16_hybrid `
  --v15-builder ..\auditor_agent_sft_v15_hq_current_package\scripts\build_v15_hq_current_dataset.py
```

See `server_scripts/RUN_SERVER.md` for training and evaluation.

Final audit results and immutable dataset hashes are recorded in
`SFT_DATA_AUDIT.md`. The machine-readable 50-sample QC report is stored beside
the dataset as `quality_audit_v16.json`.
