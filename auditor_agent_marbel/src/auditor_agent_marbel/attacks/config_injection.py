from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from auditor_agent_marbel.attacks.payloads import render_payload
from auditor_agent_marbel.core.schemas import AttackSpec


def apply_static_config_attack(
    config: Dict[str, Any],
    spec: AttackSpec,
    placement: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    attacked = deepcopy(config)
    payload = render_payload(spec, context)

    if spec.surface == "input":
        task = dict(attacked.get("task", {}))
        content = str(task.get("content", ""))
        task["content"] = f"{content}\n\n[Injected instruction]\n{payload}"
        attacked["task"] = task
    elif spec.surface == "profile":
        target = placement.get("target_agent")
        for agent in attacked.get("agents", []):
            if agent.get("agent_id") == target:
                profile = str(agent.get("profile", ""))
                agent["profile"] = f"{profile}\n\n[Injected profile]\n{payload}"
                break

    return attacked

