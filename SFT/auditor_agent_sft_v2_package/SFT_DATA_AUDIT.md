# SFT v2 Data Audit

Scope: `sft_dataset_graph_grounded_v2/` and `sft_samples_graph_grounded_v2/`.

Dataset size:

```text
all.jsonl    1460
train.jsonl  1161
test.jsonl    299
```

The SFT-visible data was checked for the following strings/fields:

```text
attack_id                  0
success_markers            0
attack_metadata            0
[Injected instruction]     0
ACI_[A-Z0-9_]+             0
END_NEGOTIATION            0
marker_check               0
offline verifier           0
attack-success index       0
```

Notes:

- The converter script may contain offline labeling field names because it reads source artifacts and redacts them before SFT export.
- The README may mention these terms to document the cleanup policy.
- The counts above refer only to SFT-visible dataset and sample files.
