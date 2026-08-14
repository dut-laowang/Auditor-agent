from __future__ import annotations

import json
from typing import Any


OBSERVABLE_TYPES = {
    "agent_act", "agent_output", "message_send", "message_receive",
    "tool_call", "tool_result", "final_result", "final_outcome", "run_end",
}


def _text(row: dict[str, Any], event_type: str) -> Any:
    if event_type == "agent_act":
        # MARBLE records the actual prompt delivered to the acting agent here.
        return row.get("content") or row.get("clean_content") or ""
    if event_type == "agent_output":
        return row.get("result") or row.get("output") or row.get("content") or ""
    if event_type in {"message_send", "message_receive"}:
        # AppWorld explicitly distinguishes the pre-injection value from the
        # framework-visible delivered value.  Only the latter is valid auditor input.
        return row.get("delivered_content") or row.get("content") or row.get("clean_content") or ""
    if event_type == "tool_call":
        return row.get("arguments") or ""
    if event_type == "tool_result":
        return row.get("delivered_content") or row.get("result") or row.get("content") or row.get("clean_content") or ""
    return row.get("result") or row.get("content") or row.get("status") or ""


def install(builder) -> None:
    def project(v15, archive, root, run_id, replacements):
        rows = v15.read_jsonl_from_zip(archive, f"{root}/data/trajectories/{run_id}.jsonl")
        projected: list[dict[str, Any]] = []
        for row in rows:
            event_type = str(row.get("event_type") or "")
            if event_type not in OBSERVABLE_TYPES:
                continue
            value = _text(row, event_type)
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            value = builder.observable_excerpt(
                builder.semantic_sanitize_text(value, replacements, run_id), 900
            )
            if not value:
                continue
            event: dict[str, Any] = {"source": "trajectory", "type": event_type, "text": value}
            if row.get("agent") or row.get("agent_id"):
                event["agent"] = str(row.get("agent") or row.get("agent_id"))
            for key in ("source_agent", "target_agent", "tool"):
                if row.get(key):
                    event[key] = str(row[key])
            projected.append(event)
        return builder.select_observable_events(projected, max_events=20)

    builder.project_observable_trajectory = project

