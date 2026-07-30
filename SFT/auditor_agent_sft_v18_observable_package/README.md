# MAS Auditor SFT V18 Observable

V18 is a minimal correction of V17. It keeps the original labels, but builds
model input only from deployment-observable task, graph, agent, message, tool,
result, and event-order evidence.

## Data

- all: 8,487
- train: 6,784
- test: 1,703
- train/test task-group overlap: 0
- excluded for insufficient visible attack evidence: 789

`train.jsonl.zip` and `all.jsonl.zip` are lossless archives used only to stay
below GitHub's single-file limit. `run_all_v18.sh` restores `train.jsonl`
automatically.

## Guarantees

- labels are preserved; no relabeling
- event selection is label-blind and type/temporal balanced
- benchmark IDs, attack flags, marker scans, judge outputs, gold labels, and
  private metadata never enter the user message
- natural-language attack instructions, propagation, tool effects, and real
  outcomes remain observable after per-run entity redaction
- 200-sample QC and leakage/proxy audits are included with the dataset

Run:

```bash
BASE=/gs/bs/tgh-26IAW/hongbo/project_4_coauthor \
REPO=$BASE/Auditor-agent GPU=0 \
bash SFT/auditor_agent_sft_v18_observable_package/server_scripts/run_all_v18.sh
```
