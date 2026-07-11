# V12-HQ-Final MAS Auditor SFT

V12-HQ-Final keeps the V12 run-level labels and train/test split, but removes low-quality localization supervision.

Core policy:

```text
Keep scale. Reduce localization noise.
```

## What Changes From V12

- Very sparse rows are filtered.
- High-confidence global/input and evidence-backed component localization are kept.
- Low-evidence component localization is downgraded:

```json
{
  "scope": "unknown",
  "component_ids": [],
  "confidence": "low"
}
```

This avoids teaching the model unsupported node/edge/tool labels.

## Dataset Size

Generated locally from V12:

```text
all:   7198 / 7362
train: 5741
test:  1457
```

Localization confidence:

```text
high:   2283
medium: 1463
low:    3452
```

## No-Leak Policy

The user-visible input does not expose:

- attack ids
- success markers
- ACI markers
- injected-instruction labels
- verifier provenance
- attack metadata

Quality checks:

```text
leak_hits = {}
invalid_trace_refs = 0
```

## Files

- `sft_dataset_graph_grounded_v12_hq_final/`: generated HQ dataset
- `server_scripts/build_v12_hq_dataset.py`: deterministic V12 -> HQ converter
- `server_scripts/run_all_v12_hq.sh`: one-shot server run
