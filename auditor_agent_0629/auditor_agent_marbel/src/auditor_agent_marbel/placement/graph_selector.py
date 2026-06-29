from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


Edge = Tuple[str, str, str]


def agent_ids(config: Dict[str, Any]) -> List[str]:
    return [str(agent["agent_id"]) for agent in config.get("agents", [])]


def relationship_edges(config: Dict[str, Any]) -> List[Edge]:
    edges: List[Edge] = []
    for item in config.get("relationships", []):
        if len(item) >= 2:
            rel = str(item[2]) if len(item) >= 3 else "collaborate with"
            edges.append((str(item[0]), str(item[1]), rel))
    return edges


def central_node(config: Dict[str, Any]) -> Optional[str]:
    ids = agent_ids(config)
    if not ids:
        return None
    degree: Counter[str] = Counter({agent_id: 0 for agent_id in ids})
    for source, target, _ in relationship_edges(config):
        degree[source] += 1
        degree[target] += 1
    return degree.most_common(1)[0][0]


def peripheral_node(config: Dict[str, Any]) -> Optional[str]:
    ids = agent_ids(config)
    if not ids:
        return None
    center = central_node(config)
    for agent_id in reversed(ids):
        if agent_id != center:
            return agent_id
    return ids[-1]


def critical_edge(config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    edges = relationship_edges(config)
    if not edges:
        ids = agent_ids(config)
        return (ids[0], ids[1]) if len(ids) >= 2 else None
    center = central_node(config)
    for source, target, _ in edges:
        if source == center or target == center:
            return source, target
    source, target, _ = edges[0]
    return source, target


def peer_edge(config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    edges = relationship_edges(config)
    center = central_node(config)
    for source, target, _ in edges:
        if source != center and target != center:
            return source, target
    if edges:
        source, target, _ = edges[-1]
        return source, target
    ids = agent_ids(config)
    return (ids[0], ids[-1]) if len(ids) >= 2 else None


def resolve_placement(config: Dict[str, Any], placement: Dict[str, Any]) -> Dict[str, Any]:
    placement_type = placement.get("type", "global")
    selector = placement.get("selector", "")
    resolved = dict(placement)

    if placement_type == "global":
        return resolved
    if placement_type == "node":
        if selector == "central_node":
            resolved["target_agent"] = central_node(config)
        elif selector == "peripheral_node":
            resolved["target_agent"] = peripheral_node(config)
        else:
            resolved["target_agent"] = placement.get("target_agent")
        return resolved
    if placement_type == "edge":
        edge = critical_edge(config) if selector == "critical_edge" else peer_edge(config)
        if edge:
            resolved["source_agent"], resolved["target_agent"] = edge
        return resolved
    if placement_type == "tool_edge":
        resolved["target_agent"] = placement.get("target_agent") or central_node(config)
        return resolved
    return resolved

