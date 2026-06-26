#!/usr/bin/env python3
"""Convert raw ACIArena mini-run JSON files into graph-augmented prefix JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_from_turn(turn: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "t": idx + 1,
        "type": "tool_call" if turn.get("tool") else "message",
        "from": turn.get("sender"),
        "to": turn.get("receiver"),
        "content": turn.get("message", ""),
        "tool": turn.get("tool"),
    }


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


def graph_view(events: list[dict[str, Any]], current: dict[str, Any], malicious_agents: list[str]) -> dict[str, Any]:
    nodes = sorted({e["from"] for e in events if e.get("from")} | {e["to"] for e in events if e.get("to")})
    edges = [f"{e['from']}->{e['to']}" for e in events if e.get("from") and e.get("to")]
    return {
        "nodes": nodes,
        "edges_so_far": edges,
        "current_edge": f"{current.get('from')}->{current.get('to')}",
        "malicious_agents": malicious_agents,
        "source_trust": "unknown",
        "receiver_tools": [],
        "privilege_boundaries": [],
    }


def weak_label_for_event(
    current: dict[str, Any],
    *,
    attack_success: bool | None,
    attack_surface: str | None,
    malicious_agents: list[str],
) -> dict[str, Any]:
    """A conservative weak label.

    This is not final paper-quality annotation. It only marks candidate unsafe
    events that should be reviewed by rules/LLM/humans later.
    """
    sender = current.get("from")
    receiver = current.get("to")
    current_edge = f"{sender}->{receiver}"
    is_candidate = bool(attack_success) and (
        sender in malicious_agents
        or receiver in malicious_agents
        or attack_surface in {"adv_input", "message_poison", "malicious_agent"}
    )
    return {
        "action": "block_candidate" if is_candidate else "allow",
        "unsafe_event": current.get("t") if is_candidate else None,
        "unsafe_edge": current_edge if is_candidate else None,
        "risk_type": [],
        "attack_surface": attack_surface,
        "intervention_point": f"before_event_{current.get('t')}" if is_candidate else None,
        "annotation_status": "weak_auto_candidate" if is_candidate else "auto_allow",
    }


def convert_one(raw: dict[str, Any]) -> list[dict[str, Any]]:
    meta = raw.get("meta_data", {})
    result = raw.get("result") or {}
    turns = raw.get("turns") or []
    events = [event_from_turn(turn, idx) for idx, turn in enumerate(turns)]
    malicious_agents = list(result.get("malicious_agents") or meta.get("malicious_agents") or [])
    agents = build_agents(events, malicious_agents)
    attack_surface = result.get("attack_surface")
    attack_success = result.get("attack_success")

    records = []
    for idx, current in enumerate(events):
        prefix = events[: idx + 1]
        records.append(
            {
                "source_dataset": "ACIArena",
                "session_id": raw.get("session_id"),
                "task": result.get("query") or meta.get("task") or "",
                "mas_framework": meta.get("mas"),
                "attack": {
                    "suite": meta.get("suite"),
                    "attack_class": result.get("attack_class"),
                    "attack_surface": attack_surface,
                    "attack_success": attack_success,
                    "attack_payload": result.get("attack_payload"),
                },
                "agents": agents,
                "event_prefix": prefix,
                "graph_view": graph_view(prefix, current, malicious_agents),
                "current_event": current,
                "label": weak_label_for_event(
                    current,
                    attack_success=attack_success,
                    attack_surface=attack_surface,
                    malicious_agents=malicious_agents,
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("plan_e/aci_min_repro/outputs/raw_runs"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("plan_e/aci_min_repro/outputs/audit_prefix_samples.jsonl"))
    args = parser.parse_args()

    records = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        records.extend(convert_one(load_json(path)))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prefix records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
