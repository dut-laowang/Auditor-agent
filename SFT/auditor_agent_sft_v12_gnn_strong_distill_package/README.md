# V12 GNN-Strong-Distill MAS Auditor

This package keeps V12 as the main SFT baseline and adds a stronger GNN teacher before SFT distillation.

It is designed to follow the relevant MAS-GNN defense ideas more closely than the lightweight package:

- G-Safeguard-style: MAS utterance graph, text embeddings, edge-aware graph attention, suspicious component scoring
- BlindGuard-style: graph-based anomalous component/risk modeling from normal and attacked MAS interactions

It is not a byte-for-byte official reproduction of either paper. It is a V12-compatible adaptation to our MARBLE SFT data and metrics.

## Core Flow

```text
V12 sanitized MAS evidence
-> strong utterance graph dataset
-> sentence-transformer node/edge embeddings
-> edge-aware GAT teacher
-> component risk ranking
-> assistant-side SFT distillation target
-> Qwen3-8B LoRA auditor
```

## No-Leak Policy

The user-visible SFT input remains V12-style sanitized input. It does not expose:

- attack ids
- success markers
- ACI markers
- injected-instruction labels
- verifier provenance
- attack metadata

The GNN teacher ranking is written only into the assistant target:

```json
"localization": {
  "scope": "edge",
  "component_ids": ["E::agent1->agent2"],
  "candidate_ranking": [
    {"id": "E::agent1->agent2", "risk": "high", "type": "edge", "target": true},
    {"id": "N::agent1", "risk": "medium", "type": "node"}
  ]
}
```

## Main Scripts

- `build_strong_gnn_graph_dataset.py`: creates utterance graph data from V12
- `train_strong_edge_gat_teacher.py`: embeds node/edge texts and trains the strong GNN teacher
- `build_distilled_sft.py`: creates distilled SFT data from V12 + GNN scores
- `train_qwen3_lora_sft.py`: Qwen3-8B LoRA SFT
- `eval_qwen3_fullschema.py`: evaluation

## Expected Outputs

```text
$GNN_OUT/metrics.json
$GNN_OUT/all_scores.jsonl
$DISTILL_DATA/train.jsonl
$DISTILL_DATA/test.jsonl
$DISTILL_50/metrics.json
$DISTILL_FULL/metrics.json
```

Use `server_scripts/run_all_v12_gnn_strong_distill.sh` for the full run.
