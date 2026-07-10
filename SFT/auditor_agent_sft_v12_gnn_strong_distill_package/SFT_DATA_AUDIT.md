# V12 GNN-Distill Data Audit

This package does not store generated SFT data in Git. The distilled dataset is generated on the server from the existing V12 dataset and graph-detector scores.

The generator enforces:

- same V12 train/test split
- same V12 marker-based labels
- no markers or attack metadata in user-visible input
- graph-detector ranking is assistant-side supervision only
- trace references must point to input evidence or graph candidates

Forbidden strings checked in user-visible input:

- `ACI_...`
- `attack_id`
- `attack_metadata`
- `success_marker`
- `success_markers`
- `[Injected instruction]`

Runtime output:

```text
$DISTILL_DATA/all.jsonl
$DISTILL_DATA/train.jsonl
$DISTILL_DATA/test.jsonl
$DISTILL_DATA/stats.json
```

The expected clean check in `stats.json` is:

```json
{
  "leak_check": {
    "all": {},
    "train": {},
    "test": {}
  },
  "invalid_trace_refs": {
    "all": 0,
    "train": 0,
    "test": 0
  }
}
```
