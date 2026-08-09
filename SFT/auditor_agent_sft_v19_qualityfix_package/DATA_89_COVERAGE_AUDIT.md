# `data-89` source coverage audit

## Decision

The two archives in `plan_e/data-89` do not fully replace the V18 source
bundle by themselves.

| Source | Framework / topology | Single attacked | Dual attacked | Role in V19 |
|---|---|---:|---:|---|
| legacy bundle | MARBLE chain/graph/star/tree | 6,840 | 1,200 | retain chain/graph/tree only |
| `weekend_fresh_star_fixed.zip` | MARBLE star | 1,710 | 300 | replace legacy star |
| `autogen_native_minimal_20260806.tar.zst` | AutoGen selector-group-chat/swarm | 3,420 | 600 | add cross-framework coverage |

The new star archive is exactly one-topology coverage; the legacy bundle has
four MARBLE topologies. The AutoGen README explicitly says that all MARBLE data
are excluded, so AutoGen cannot stand in for chain/graph/tree.

## V19 assembly result

- legacy MARBLE non-star selected: 6,638
- fixed MARBLE star selected: 1,209
- AutoGen selected: 1,785
- combined: 9,632
- train: 5,588
- validation: 2,216
- sealed test: 1,828

AutoGen is a minimal analysis package: generated configs, raw model results,
logs, and judge caches are excluded. Its exact task instruction is therefore
not recoverable from config; the observable trajectory remains available.
Private-control rows lacking a supplied private-control signal are conservatively
excluded rather than assumed clean. Attacked rows failing the inherited V18
observable-evidence threshold also remain excluded and are counted in per-source
audit artifacts.

## Integrity result

- source/run overlap across splits: 0
- `(scenario, sample_id)` overlap across splits: 0
- exact and normalized input overlap across splits: 0
- forbidden privileged input keys/literals: 0
- invalid component/evidence references: 0
- dual attacked samples using the old global fallback: 0
- final-test model evaluations performed during construction: 0

The validation-only lexical proxy reaches 71.98% accuracy (macro-F1 70.91%),
so shortcut and counterfactual ablations remain mandatory for the V19 report.
