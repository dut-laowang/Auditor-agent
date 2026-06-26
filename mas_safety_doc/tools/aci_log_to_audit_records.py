#!/usr/bin/env python3
"""Convert an ACIArena logger JSON file into graph-augmented prefix records.

This script only uses fields that ACIArena's MASLogger actually writes:
meta_data, turns[].sender, turns[].receiver, turns[].message, turns[].tool, result.

Attack-level labels such as attack_success or unsafe_edge are optional inputs
because the current ACIArena logger JSON does not bind them to each trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def event_from_turn(turn: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "t": idx + 1,
        "type": "tool_call" if turn.get("tool") else "message",
        "from": turn.get("sender"),
        "to": turn.get("receiver"),
        "content": turn.get("message", ""),
        "tool": turn.get("tool"),
    }


def infer_attack_surface(attack_class: str | None) -> str | None:
    if not attack_class:
        return None
    name = attack_class.lower()
    if "message" in name or "poison" in name:
        return "message_poison"
    if "agent" in name:
        return "malicious_agent"
    if "instruction" in name or "input" in name:
        return "adv_input"
    return None


def build_agents(events: list[dict[str, Any]], malicious_agents: list[str]) -> list[dict[str, Any]]:
    ids = sorted({e["from"] for e in events if e.get("from")} | {e["to"] for e in events if e.get("to")})
    return [
        {
            "id": agent_id,
            "role": agent_id,
            "tools": [],
            "privilege": "unknown",
            "is_malicious_target": agent_id in malicious_agents,
        }
        for agent_id in ids
    ]


def graph_view_for_prefix(events: list[dict[str, Any]], current: dict[str, Any], malicious_agents: list[str]) -> dict[str, Any]:
    nodes = sorted({e["from"] for e in events if e.get("from")} | {e["to"] for e in events if e.get("to")})
    edges = [f"{e['from']}->{e['to']}" for e in events if e.get("from") and e.get("to")]
    return {
        "nodes": nodes,
        "edges_so_far": edges,
        "current_edge": f"{current.get('from')}->{current.get('to')}",
        "malicious_agents": malicious_agents,
        "receiver_tools": [],
        "privilege_boundaries": [],
        "source_trust": "unknown",
    }


def convert_session(
    session: dict[str, Any],
    *,
    attack_class: str | None,
    attack_success: bool | None,
) -> list[dict[str, Any]]:
    meta = session.get("meta_data", {})
    turns = session.get("turns", [])
    events = [event_from_turn(turn, idx) for idx, turn in enumerate(turns)]
    malicious_agents = list(meta.get("malicious_agents") or [])
    agents = build_agents(events, malicious_agents)
    attack_surface = infer_attack_surface(attack_class)

    records = []
    for idx, current in enumerate(events):
        prefix = events[: idx + 1]
        records.append(
            {
                "source_dataset": "ACIArena",
                "task": meta.get("task") or meta.get("query") or "",
                "mas_framework": meta.get("mas"),
                "attack": {
                    "suite": meta.get("suite"),
                    "attack_class": attack_class,
                    "attack_surface": attack_surface,
                    "attack_success": attack_success,
                },
                "agents": agents,
                "event_prefix": prefix,
                "graph_view": graph_view_for_prefix(prefix, current, malicious_agents),
                "current_event": current,
                "label": {
                    "action": "unknown",
                    "unsafe_event": None,
                    "unsafe_edge": None,
                    "risk_type": [],
                    "intervention_point": None,
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--attack-class", default=None)
    parser.add_argument("--attack-success", choices=["true", "false", "unknown"], default="unknown")
    args = parser.parse_args()

    attack_success = None
    if args.attack_success != "unknown":
        attack_success = args.attack_success == "true"

    session = json.loads(args.input_json.read_text(encoding="utf-8"))
    records = convert_session(session, attack_class=args.attack_class, attack_success=attack_success)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
