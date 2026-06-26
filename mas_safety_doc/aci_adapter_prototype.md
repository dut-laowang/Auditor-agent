# ACIArena -> Graph-Augmented Prefix Adapter Prototype

This note records what can be implemented from the current local ACIArena codebase without inventing unavailable fields.

## Verified Raw Fields

ACIArena `MASLogger` saves:

```json
{
  "session_id": "...",
  "meta_data": {...},
  "turns": [
    {
      "sender": "...",
      "receiver": "...",
      "message": "...",
      "tool": null
    }
  ],
  "result": null
}
```

These fields are enough to construct a temporal event prefix and a minimal graph view:

```text
nodes = unique(sender, receiver)
edges_so_far = sender -> receiver for all events in current prefix
current_edge = current_event.sender -> current_event.receiver
```

## Fields That Need Extra Instrumentation

ACIArena currently computes attack success in `attack.verify()`, but the default logger JSON does not bind each logged trajectory to:

```text
attack_class
attack_surface
attack_success
task_id
utility_success
```

For SFT data construction, the executor should save one record per `(task, attack)` run:

```json
{
  "source": "ACIArena",
  "task": "...",
  "mas": "...",
  "suite": "hijacking|disruption|disclosure",
  "attack_class": "...",
  "attack_surface": "adv_input|malicious_agent|message_poison",
  "malicious_agents": ["..."],
  "turns": [...],
  "attack_success": true,
  "utility_success": false
}
```

## Adapter Output Schema

Each current event becomes one prefix sample:

```json
{
  "task": "...",
  "mas_framework": "...",
  "attack": {
    "suite": "...",
    "attack_class": "...",
    "attack_surface": "...",
    "attack_success": true
  },
  "agents": [
    {"id": "...", "role": "...", "tools": [], "privilege": "unknown"}
  ],
  "event_prefix": [
    {"t": 1, "from": "...", "to": "...", "type": "message", "content": "..."}
  ],
  "graph_view": {
    "nodes": ["..."],
    "edges_so_far": ["A->B"],
    "current_edge": "A->B",
    "malicious_agents": ["..."],
    "receiver_tools": [],
    "privilege_boundaries": []
  },
  "current_event": {
    "t": 1,
    "from": "...",
    "to": "...",
    "type": "message",
    "content": "..."
  },
  "label": {
    "action": "allow|block|warn",
    "unsafe_event": null,
    "unsafe_edge": null,
    "risk_type": [],
    "intervention_point": null
  }
}
```

## What Is Realistic Now

Minimal implementation is feasible:

```text
ACIArena logger turns
        ↓
timestamped events
        ↓
graph_view from sender/receiver edges
        ↓
prefix samples
```

But high-quality training labels are not automatic yet:

```text
unsafe_event / unsafe_edge
privilege_boundary
source_trust
instruction_laundering
```

These require rules, LLM judge, or additional instrumentation.

