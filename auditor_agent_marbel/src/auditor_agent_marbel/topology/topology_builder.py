from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def _agent_ids(config: Dict[str, Any]) -> List[str]:
    return [str(agent["agent_id"]) for agent in config.get("agents", [])]


def apply_topology(sample: Dict[str, Any], topology: str) -> Dict[str, Any]:
    config = deepcopy(sample)
    ids = _agent_ids(config)
    if not ids:
        raise ValueError("Sample has no agents")

    config["coordinate_mode"] = topology

    if topology == "graph":
        config["relationships"] = config.get("relationships", [])
    elif topology == "star":
        center = ids[0]
        relationships = []
        for agent_id in ids[1:]:
            relationships.append([center, agent_id, "collaborate with"])
            relationships.append([agent_id, center, "collaborate with"])
        config["relationships"] = relationships
    elif topology == "tree":
        root = ids[0]
        config["relationships"] = [[root, agent_id, "parent"] for agent_id in ids[1:]]
    else:
        raise ValueError(f"Unsupported pilot topology: {topology}")

    return config

