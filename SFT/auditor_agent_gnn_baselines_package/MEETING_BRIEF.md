# Meeting Brief: V12 SFT Auditor

## 1. SFT JSON Structure

V12 is a graph-grounded instruction-tuned auditor. The model input is MAS task + graph + sanitized run evidence + graph candidates. The model output is a structured audit JSON.

核心区别：输入只给未知 MAS run 的图结构与脱敏运行证据，不提前暴露 attack id、marker、surface、objective 或 attack location；输出统一给出 verdict、attack type、component-level localization 和 evidence-linked audit trace。

Note: `_note` fields below are display comments only; they are not included in the actual SFT JSONL data.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a graph-grounded MAS safety auditor..."
    },
    {
      "role": "user",
      "content": {
        "schema": "Graph-grounded-Candidate-SFT/v12",
        "_note": "输入侧：只描述任务、MAS graph 和脱敏运行证据；不提供攻击答案。",
        "task": {
          "scenario": "research / coding / bargaining / minecraft",
          "goal": "original MAS task"
        },
        "graph": {
          "_note": "显式给出 MAS 拓扑，支持 chain/star/tree/graph 等结构。",
          "topology": "chain / star / tree / graph",
          "nodes": ["agent1", "agent2", "agent3"],
          "edges": [
            {"source": "agent1", "target": "agent2", "relation": "collaborates_with"}
          ]
        },
        "audit_request": {
          "mode": "unknown_run_audit",
          "_note": "未知运行审计：模型需要自己判断 clean、攻击失败或攻击成功。"
        },
        "run_evidence": {
          "_note": "脱敏轨迹和最终输出；marker 与注入标签已清理。",
          "observed": [
            {"id": "obs_0", "type": "agent_act", "agent": "agent1", "text": "sanitized run evidence"}
          ],
          "final_output": [
            {"id": "out_0", "type": "final_outcome", "text": "sanitized final output"}
          ],
          "reference": {"available": false, "clean": []}
        },
        "graph_candidates_note": "候选定位空间：global / node / edge / tool，便于统一定位输出。",
        "graph_candidates": [
          {"id": "G::run", "type": "global"},
          {"id": "N::agent1", "type": "node", "agent": "agent1"},
          {"id": "E::agent1->agent2", "type": "edge", "source": "agent1", "target": "agent2"},
          {"id": "T::agent1", "type": "tool", "agent": "agent1"}
        ]
      }
    },
    {
      "role": "assistant",
      "content": {
        "_note": "输出侧：先判定安全性，再给攻击类型、定位和证据引用。",
        "decision": {
          "verdict": "clean_safe / attack_failed / attack_success",
          "binary_label": "safe / unsafe"
        },
        "attack": {
          "present": true,
          "surface": "input / memory / message / profile / tool / dual / none",
          "objective": "disclosure / disruption / hijacking / none"
        },
        "localization": {
          "_note": "用 graph candidate id 表示定位结果，而不是自由文本猜测。",
          "scope": "global / node / edge / tool",
          "component_ids": ["N::agent2"]
        },
        "audit_trace": [
          {
            "step": "localize_component",
            "component_refs": ["N::agent2"],
            "evidence_refs": ["obs_0"]
          },
          {
            "step": "verify_outcome_effect",
            "component_refs": ["N::agent2"],
            "evidence_refs": ["out_0"]
          }
        ]
      }
    }
  ]
}
```

Key point: V12 is not a GNN. It is `Qwen3-8B + LoRA SFT` with graph-grounded input/output schema.

## 2. Dataset Scale

| Split | Total | clean_safe | attack_failed | attack_success |
|---|---:|---:|---:|---:|
| train | 5,875 | 144 | 2,773 | 2,958 |
| test | 1,487 | 48 | 700 | 739 |
| all | 7,362 | 192 | 3,473 | 3,697 |

| Split | input | memory | message | profile | tool | dual | none |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,044 | 1,037 | 1,065 | 1,064 | 1,067 | 454 | 144 |
| test | 247 | 256 | 275 | 273 | 273 | 115 | 48 |
| all | 1,291 | 1,293 | 1,340 | 1,337 | 1,340 | 569 | 192 |

Quality checks:

| Check | Result |
|---|---:|
| leak check for marker/attack metadata in SFT input | passed |
| invalid evidence refs | 0 |
| avg graph candidates | 13.435 |
| candidate types | global / node / edge / tool |

## 3. Common50 Result Summary

Test set: balanced common50, `safe=31`, `unsafe=19`.

### strict_agent

Only `N::agentX` is counted as an agent label.

| Method | Binary Acc | Safe F1 | Unsafe F1 | Agent P | Agent R | Agent F1 | Agent Hit | Top1 | Top2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V8-SFT | 78.0 | 80.0 | 75.6 | 5.6 | 15.4 | 8.2 | 18.2 | N/A | N/A |
| V12-SFT | 76.0 | 79.3 | 71.4 | 38.5 | 38.5 | 38.5 | 36.4 | N/A | N/A |
| G-Safeguard | 70.0 | 79.5 | 44.4 | 35.3 | 46.2 | 40.0 | 45.5 | 36.4 | 72.7 |
| BlindGuard | 44.0 | 17.6 | 57.6 | 12.3 | 61.5 | 20.5 | 63.6 | 63.6 | 72.7 |

### agent_or_tool_owner

`N::agentX` and `T::agentX` are counted as agent labels. `E::agentA->agentB` and `G::run` are not counted as agents.

| Method | Binary Acc | Safe F1 | Unsafe F1 | Agent P | Agent R | Agent F1 | Agent Hit | Top1 | Top2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V8-SFT | 78.0 | 80.0 | 75.6 | 8.3 | 21.4 | 12.0 | 25.0 | N/A | N/A |
| V12-SFT | 76.0 | 79.3 | 71.4 | 28.0 | 50.0 | 35.9 | 50.0 | N/A | N/A |
| G-Safeguard | 70.0 | 80.0 | 40.0 | 35.7 | 35.7 | 35.7 | 33.3 | 58.3 | 75.0 |
| BlindGuard | 42.0 | 17.1 | 55.4 | 13.1 | 57.1 | 21.3 | 58.3 | 58.3 | 66.7 |

## Takeaway

V12 is the current main version. V8 has stronger run-level binary performance but weak localization. V12 keeps competitive run-level auditing and improves projected agent localization, while preserving full audit output: verdict, attack type, graph component localization, and evidence trace. GNN baselines are useful for agent-level comparison, but unsupported audit metrics are marked as `N/A`.
