# Frozen comparison protocol

This protocol is fixed before baseline predictions are generated.

## Primary controlled comparison

Compare V18-Flat and V18-Graph on the exact same 1,703 test run IDs.

- Primary classification endpoint: three-class macro-F1.
- Secondary classification endpoints: three-class accuracy, binary accuracy,
  unsafe precision, and unsafe recall.
- Attribution endpoints: component micro-F1 and scope accuracy.
- Prespecified slices: single/dual, topology, surface, and gold component scope.

Both use Qwen3-8B, two epochs, LoRA rank 16, learning rate 2e-4, effective
batch size 16, max length 6144, deterministic generation, and identical target
reports. Flat and Graph differ only in explicit graph representation and system
instruction. Observable event content and order are byte-equivalent.

## External trajectory baseline

Run the frozen `AI45Research/AgentDoG1.5-Llama-3.1-8B` checkpoint on the exact
same V18 200-run subset.

- Official action-safety protocol: secondary transfer diagnostic because its
  native `unsafe` definition is not equivalent to V18 `attack_success`.
- Outcome-adapted protocol: primary AgentDoG-to-V18 binary comparison.
- No few-shot examples, calibration, threshold selection, or prompt revision
  after predictions are observed.

## Interpretation

AgentDoG alone cannot establish the value of graph structure because its
training corpus and base model differ. The causal evidence for graph structure
is the controlled V18-Flat versus V18-Graph comparison. AgentDoG measures
external transfer of an advanced trajectory guardrail.
