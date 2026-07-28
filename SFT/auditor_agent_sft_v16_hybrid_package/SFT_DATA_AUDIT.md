# V16-Hybrid SFT Data Audit

## Final status

- Build validation: passed (`validation = {}`)
- Source JSONL parse errors: Single 0, Dual 0
- Train/test task-group overlap: 0
- Dynamic marker/private-value hits in visible messages: 0
- Static forbidden-pattern hits: 0
- Invalid evidence, component, or localization references: 0

## Source-label preservation

The 8,040 attacked labels are read from the supplied final-label files and are
not re-judged or rewritten:

- `success`: 2,510 -> `attack_success`
- `failure`: 4,112 -> `attack_failed`
- `not_exposed`: 1,268 -> retained in the exclusion audit
- `ambiguous`: 142 -> retained in the exclusion audit
- `invalid`: 8 -> covered by the non-completed exclusion audit

The training set additionally uses eligible controls:

- eligible private controls: 2,494
- completed clean controls: 160

## Final dataset

- all: 9,276
- train: 7,421
- test: 1,855
- `clean_safe`: 2,654
- `attack_failed`: 4,112
- `attack_success`: 2,510

SHA-256:

- `all.jsonl`: `649b76ec96d87dadd99c4168ebf8f077feed951015365a03b87a31409081f86c`
- `train.jsonl`: `7e5d0abb04c7ffa83e440423e47a5210eb39f2345961873fbe594516782baace`
- `test.jsonl`: `eb8b91d197cf25e9476936629000583f4ce948c2006f19500e43cd961358938a`

## Fifty-sample quality control

The deterministic sample is stratified across scenario, single/dual attack
mode, and target verdict. All 50 passed:

- three-message SFT structure: 50/50
- source-label to target mapping: 50/50
- visible run evidence: 50/50
- graph candidates present: 50/50
- assistant target is valid JSON: 50/50

The machine-readable audit, complete strata, and file hashes are in
`sft_dataset_graph_grounded_v16_hybrid/quality_audit_v16.json`.

## Evaluation scope

The one-shot server workflow produces:

- three-class accuracy/report;
- binary accuracy/report;
- localization component precision, recall, F1, hit rate, exact match, and
  scope accuracy;
- audit-trace JSON/reference-quality metrics;
- V12 versus V16 results on the exact packaged V12 common-50 test set.

Localization remains a projection of source attack placement and is not
presented as independently re-judged causal attribution.
