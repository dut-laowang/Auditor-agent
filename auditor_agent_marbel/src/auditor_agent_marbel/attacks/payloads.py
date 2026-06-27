from __future__ import annotations

from typing import Any, Dict

from auditor_agent_marbel.core.schemas import AttackSpec


def render_payload(spec: AttackSpec, context: Dict[str, Any]) -> str:
    payload = spec.payload_template
    replacements = {
        "scenario": context.get("scenario", ""),
        "sample_id": context.get("sample_id", ""),
        "topology": context.get("topology", ""),
        "attack_id": spec.attack_id,
        "objective": spec.objective,
        "surface": spec.surface,
    }
    for key, value in replacements.items():
        payload = payload.replace("{" + key + "}", str(value))
    return payload

