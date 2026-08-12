from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "Graph-grounded-Candidate-SFT/v13"
STRUCTURAL_MARKER_ALLOWLIST = {"audit_trace"}
SOURCE_LABEL_TO_VERDICT = {
    "success": ("attack_success", "unsafe"),
    "failure": ("attack_failed", "safe"),
}
EVIDENCE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "as", "is", "are", "be", "this", "that", "when", "from", "than", "it",
    "its", "their", "your", "you", "include", "final", "into", "must", "should",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_v15_builder(path: Path):
    spec = importlib.util.spec_from_file_location("v15_builder_for_v18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V15 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectoryArchive:
    """Read-only adapter exposing a final/merged directory as the V15 virtual ZIP layout."""

    def __init__(
        self,
        merged_dir: Path,
        virtual_root: str,
        *,
        trajectories_dir: Path | None = None,
        configs_dir: Path | None = None,
        results_dir: Path | None = None,
    ):
        self.merged_dir = merged_dir
        self.virtual_root = virtual_root
        self.kind_dirs = {
            "configs": configs_dir or merged_dir / "configs",
            "trajectories": trajectories_dir or merged_dir / "trajectories",
            "marble_results": results_dir or merged_dir / "marble_results",
        }
        # A source file is read several times by the inherited V15 extraction
        # helpers. Track physical line locations so one malformed line is
        # counted once, rather than once per read.
        self.malformed_line_keys: set[tuple[str, int]] = set()
        self._names = [f"{virtual_root}/data/run_manifest.jsonl"]
        for kind, base in self.kind_dirs.items():
            if not base.is_dir():
                continue
            self._names.extend(
                f"{virtual_root}/data/{kind}/{path.name}"
                for path in base.iterdir()
                if path.is_file()
            )

    def namelist(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        prefix = f"{self.virtual_root}/data/"
        if not name.startswith(prefix):
            raise KeyError(name)
        relative = name[len(prefix) :]
        parts = relative.split("/", 1)
        if len(parts) == 2 and parts[0] in self.kind_dirs:
            path = self.kind_dirs[parts[0]] / parts[1]
        else:
            path = self.merged_dir / relative
        if not path.is_file():
            raise KeyError(name)
        return path.read_bytes()


def tolerant_archive_jsonl(archive: DirectoryArchive, name: str) -> list[dict[str, Any]]:
    try:
        raw = archive.read(name)
    except KeyError:
        return []
    rows = []
    # JSON strings may legally contain Unicode line/paragraph separators.
    # str.splitlines() treats those characters as record boundaries, while
    # JSONL is delimited only by physical LF bytes.
    for line_no, line in enumerate(
        raw.decode("utf-8-sig", "replace").split("\n"), 1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            archive.malformed_line_keys.add((name, line_no))
    return rows


def string_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(string_leaves(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(string_leaves(item))
        return out
    return [value] if isinstance(value, str) else []


def redaction_values(manifest_row: dict[str, Any], signal: dict[str, Any] | None) -> list[str]:
    metadata = manifest_row.get("attack_metadata") or {}
    values = list(metadata.get("success_markers") or [])
    values.extend((signal or {}).get("markers") or [])
    values.extend(string_leaves(metadata.get("private_user_information") or {}))
    # Avoid replacing common short words while ensuring concrete canaries/secrets are removed.
    return sorted({str(v) for v in values if len(str(v).strip()) >= 6}, key=len, reverse=True)


def group_split(
    rows: list[dict[str, Any]], seed: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, list[int]]],
]:
    """Create task-grouped train/validation/test splits across every framework.

    A scenario/task pair is assigned once, so the same underlying task can never
    cross a split through another topology or framework. Two task IDs are sealed
    for final test and two different IDs are used for validation per scenario.
    """

    rng = random.Random(seed)
    scenarios = sorted({str(r["metadata"]["scenario"]) for r in rows})
    held_out: dict[str, dict[str, list[int]]] = {}
    for scenario in scenarios:
        ids = sorted(
            {
                int(r["metadata"]["sample_id"])
                for r in rows
                if r["metadata"]["scenario"] == scenario
                and r["metadata"].get("sample_id") is not None
            }
        )
        shuffled = ids[:]
        rng.shuffle(shuffled)
        if len(shuffled) < 5:
            raise ValueError(f"Need at least five task groups for {scenario}, got {ids}")
        held_out[scenario] = {
            "test": sorted(shuffled[:2]),
            "validation": sorted(shuffled[2:4]),
        }

    train, validation, test = [], [], []
    for row in rows:
        meta = row["metadata"]
        sid = int(meta["sample_id"])
        assignment = held_out[str(meta["scenario"])]
        if sid in assignment["test"]:
            test.append(row)
        elif sid in assignment["validation"]:
            validation.append(row)
        else:
            train.append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test, held_out


def stable_uid(sample: dict[str, Any], uid: int) -> None:
    new_uid = f"v19_{uid:07d}"
    sample["metadata"]["sample_uid"] = new_uid
    user = json.loads(sample["messages"][1]["content"])
    user["sample_uid"] = new_uid
    sample["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)


def entity_replacements(run_id: str, secrets: list[str]) -> dict[str, str]:
    """Create stable-within-run, unlinkable-across-run replacements."""

    out: dict[str, str] = {}
    for secret in secrets:
        if secret in STRUCTURAL_MARKER_ALLOWLIST:
            continue
        digest = hashlib.sha256(f"{run_id}\0{secret}".encode("utf-8")).hexdigest()[:10]
        out[secret] = f"[REDACTED_ENTITY_{digest.upper()}]"
    return out


def semantic_sanitize_text(
    value: Any, replacements: dict[str, str], run_id: str
) -> str:
    """Remove benchmark identity while preserving ordinary attack semantics."""

    text = str(value or "")
    for secret, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if secret:
            text = text.replace(secret, replacement)

    def replace_unknown_marker(match: re.Match[str]) -> str:
        raw = match.group(0)
        digest = hashlib.sha256(f"{run_id}\0{raw}".encode("utf-8")).hexdigest()[:10]
        return f"[REDACTED_ENTITY_{digest.upper()}]"

    # Do not use a trailing word boundary: several source identifiers end in
    # underscores, for which ``\b`` does not match.
    text = re.sub(r"ACI_[A-Z0-9_]+", replace_unknown_marker, text, flags=re.I)

    # Preserve the instruction body. Only provenance tags that reveal the
    # benchmark injection mechanism are removed.
    text = re.sub(
        r"\[(?:Injected|External)(?:[^\]]*)\]\s*:?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:Injected|External)\s+(?:message|instruction|profile)\b"
        r"(?:\s+instruction)?\s*:?\s*",
        "",
        text,
        flags=re.I,
    )
    # Narrative summaries may refer to the source as "injected instructions".
    # Normalize that provenance wording without removing the described action.
    text = re.sub(
        r"\binjected\s+instructions?\b",
        "additional instructions",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\binjected\s+messages?\b",
        "additional messages",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\binjected\s+profiles?\b",
        "additional profiles",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bEND_NEGOTIATION\b",
        "terminate the negotiation",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[Private user information\]\s*:?\s*",
        "Protected context: ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bprivate_user_information\b",
        "protected context",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bPrivate user information\b",
        "Protected context",
        text,
        flags=re.I,
    )
    text = re.sub(r"Benchmark addendum", "Additional instruction", text, flags=re.I)
    text = re.sub(r"\battack[_ -]?id\b", "instruction reference", text, flags=re.I)
    text = re.sub(r"\bsuccess[_ -]?markers?\b", "tracked entity", text, flags=re.I)
    return text


def sanitize_visible_value(
    value: Any, replacements: dict[str, str], run_id: str
) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_visible_value(item, replacements, run_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_visible_value(item, replacements, run_id) for item in value
        ]
    if isinstance(value, str):
        return semantic_sanitize_text(value, replacements, run_id)
    return value


def sanitize_sample_messages(
    sample: dict[str, Any], replacements: dict[str, str], run_id: str
) -> None:
    for message in sample["messages"]:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            structured = json.loads(content)
        except json.JSONDecodeError:
            message["content"] = semantic_sanitize_text(
                content, replacements, run_id
            )
        else:
            structured = sanitize_visible_value(structured, replacements, run_id)
            serialized = json.dumps(structured, ensure_ascii=False)
            # Some tool outputs contain nested JSON encoded as strings. Apply the
            # provenance-only substitutions once more to the serialized envelope.
            message["content"] = semantic_sanitize_text(serialized, {}, run_id)


def semantic_events_from_trajectory(
    v15,
    archive: DirectoryArchive,
    root: str,
    run_id: str,
    markers: list[str],
    objective: str | None,
) -> list[dict[str, Any]]:
    """Retain injected instruction bodies instead of preferring clean_content."""

    out: list[dict[str, Any]] = []
    rows = v15.read_jsonl_from_zip(
        archive, f"{root}/data/trajectories/{run_id}.jsonl"
    )
    for row in rows:
        event_type = row.get("event_type")
        if event_type in {"run_start", "config_prepared", "attack_injected"}:
            continue
        event: dict[str, Any] = {
            "source": "trajectory",
            "type": event_type or "event",
        }
        if event_type == "agent_act":
            text = row.get("content") or row.get("clean_content") or ""
            if row.get("agent"):
                event["agent"] = row["agent"]
        elif event_type in {"message_send", "message_receive"}:
            text = (
                row.get("content")
                or row.get("message")
                or row.get("clean_content")
                or ""
            )
            for key in ("source_agent", "target_agent", "agent"):
                if row.get(key):
                    event[key] = row[key]
        elif event_type == "tool_call":
            text = row.get("arguments") or ""
            if row.get("agent"):
                event["agent"] = row["agent"]
            if row.get("tool"):
                event["tool"] = row["tool"]
        elif event_type == "tool_result":
            text = (
                row.get("result")
                or row.get("content")
                or row.get("clean_content")
                or ""
            )
            if row.get("agent"):
                event["agent"] = row["agent"]
            if row.get("tool"):
                event["tool"] = row["tool"]
        elif event_type in {"run_end", "final_result", "final_outcome"}:
            text = row.get("result") or row.get("content") or row.get("status") or ""
        else:
            continue
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        text = v15.truncate(v15.sanitize_text(text, markers, objective))
        if text:
            event["text"] = text
            out.append(event)
        if len(out) >= 8:
            break
    return out


OBSERVABLE_EVENT_TYPES = {
    "agent_act",
    "agent_output",
    "message_send",
    "message_receive",
    "tool_call",
    "tool_result",
    "final_result",
    "final_outcome",
    "run_end",
}


def _uniform_pick(indices: list[int], count: int) -> list[int]:
    """Choose temporal anchors without using labels or attack metadata."""

    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    positions = {
        round(i * (len(indices) - 1) / (count - 1))
        for i in range(count)
    }
    return [indices[i] for i in sorted(positions)]


def select_observable_events(
    events: list[dict[str, Any]], max_events: int = 20
) -> list[dict[str, Any]]:
    """Label-blind temporal/type-balanced selection over observable events."""

    families = {
        "agent": [
            i
            for i, event in enumerate(events)
            if event["type"] in {"agent_act", "agent_output"}
        ],
        "message": [
            i
            for i, event in enumerate(events)
            if event["type"] in {"message_send", "message_receive"}
        ],
        "tool": [
            i
            for i, event in enumerate(events)
            if event["type"] in {"tool_call", "tool_result"}
        ],
        "outcome": [
            i
            for i, event in enumerate(events)
            if event["type"] in {"final_result", "final_outcome", "run_end"}
        ],
    }
    selected: set[int] = set()
    for family, quota in (("agent", 6), ("message", 7), ("tool", 5), ("outcome", 2)):
        selected.update(_uniform_pick(families[family], quota))

    if len(selected) < min(max_events, len(events)):
        remaining = [i for i in range(len(events)) if i not in selected]
        selected.update(
            _uniform_pick(remaining, min(max_events - len(selected), len(remaining)))
        )
    return [events[i] for i in sorted(selected)[:max_events]]


def observable_excerpt(value: str, limit: int = 900) -> str:
    """Preserve beginning, middle, and end uniformly for every event."""

    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    third = limit // 3
    midpoint = len(text) // 2
    middle_start = max(0, midpoint - third // 2)
    return (
        text[:third]
        + " ...[middle]... "
        + text[middle_start : middle_start + third]
        + " ...[end]... "
        + text[-third:]
    )


def project_observable_trajectory(
    v15,
    archive: DirectoryArchive,
    root: str,
    run_id: str,
    replacements: dict[str, str],
) -> list[dict[str, Any]]:
    """Strict allowlist projection: no experimental fields are copied."""

    projected: list[dict[str, Any]] = []
    rows = v15.read_jsonl_from_zip(
        archive, f"{root}/data/trajectories/{run_id}.jsonl"
    )
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type not in OBSERVABLE_EVENT_TYPES:
            continue
        event: dict[str, Any] = {"source": "trajectory", "type": event_type}
        text: Any = ""
        if event_type in {"agent_act", "agent_output"}:
            text = row.get("content") or row.get("output") or row.get("clean_content") or ""
            if row.get("agent"):
                event["agent"] = str(row["agent"])
        elif event_type in {"message_send", "message_receive"}:
            text = row.get("content") or row.get("message") or row.get("clean_content") or ""
            for key in ("source_agent", "target_agent", "agent"):
                if row.get(key):
                    event[key] = str(row[key])
        elif event_type == "tool_call":
            text = row.get("arguments") or ""
            if row.get("agent"):
                event["agent"] = str(row["agent"])
            if row.get("tool"):
                event["tool"] = str(row["tool"])
        elif event_type == "tool_result":
            text = row.get("result") or row.get("content") or row.get("clean_content") or ""
            if row.get("agent"):
                event["agent"] = str(row["agent"])
            if row.get("tool"):
                event["tool"] = str(row["tool"])
        else:
            text = row.get("result") or row.get("content") or row.get("status") or ""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        text = observable_excerpt(
            semantic_sanitize_text(text, replacements, run_id), 600
        )
        if text:
            event["text"] = text
            projected.append(event)
    return select_observable_events(projected)


def observable_task_and_final(
    v15,
    archive: DirectoryArchive,
    root: str,
    run_id: str,
    replacements: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """Read only task and iteration result fields from MARBLE artifacts."""

    rows = v15.read_jsonl_from_zip(
        archive, f"{root}/data/marble_results/{run_id}.jsonl"
    )
    if not rows:
        # AutoGen minimal packages omit MARBLE result/config artifacts. Use only
        # observable agent outputs; never consume attack/label instrumentation.
        trajectory = v15.read_jsonl_from_zip(
            archive, f"{root}/data/trajectories/{run_id}.jsonl"
        )
        results: list[str] = []
        for event in trajectory:
            if event.get("event_type") != "agent_output":
                continue
            value = event.get("output") or event.get("content") or event.get("clean_content")
            if not isinstance(value, str):
                continue
            text = observable_excerpt(
                semantic_sanitize_text(value, replacements, run_id), 1000
            )
            if text and (not results or text != results[-1]):
                results.append(text)
        finals = [
            {"source": "trajectory", "type": "final_outcome", "text": text}
            for text in results[-2:]
        ]
        return "", finals
    row = rows[0]
    task = semantic_sanitize_text(row.get("task") or "", replacements, run_id)
    results: list[str] = []
    for iteration in row.get("iterations") or []:
        if not isinstance(iteration, dict) or not isinstance(iteration.get("result"), str):
            continue
        text = semantic_sanitize_text(iteration["result"], replacements, run_id)
        text = observable_excerpt(text, 1000)
        if text and (not results or text != results[-1]):
            results.append(text)
    finals = [
        {"source": "marble_result", "type": "final_outcome", "text": text}
        for text in results[-2:]
    ]
    return v15.truncate(task, 700), finals


def component_ids_from_placement(v15, row: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Project both single- and dual-site placements into graph candidates."""

    placement = row.get("placement") or {}
    if placement.get("type") != "dual":
        return v15.component_ids_from_placement(placement, row.get("surface"), candidates)

    projected: list[str] = []
    scopes: list[str] = []
    for component in placement.get("components") or []:
        component_placement = dict(component.get("placement") or {})
        for key in ("source_agent", "target_agent"):
            if component.get(key) and not component_placement.get(key):
                component_placement[key] = component[key]
        scope, ids = v15.component_ids_from_placement(
            component_placement, component.get("surface"), candidates
        )
        scopes.append(scope)
        projected.extend(ids)
    projected = list(dict.fromkeys(projected))
    if not projected:
        return "global", []
    unique_scopes = sorted(set(scopes))
    return (unique_scopes[0] if len(unique_scopes) == 1 else "multi"), projected


def aligned_evidence_refs(
    candidates: list[dict[str, Any]],
    component_ids: list[str],
    observed: list[dict[str, Any]],
    final_output: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Reference localized candidate events and observable outcome events."""

    by_id = {candidate["id"]: candidate for candidate in candidates}
    location_refs: list[str] = []
    for component_id in component_ids:
        candidate = by_id.get(component_id) or {}
        for key in ("event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"):
            location_refs.extend(candidate.get(key) or [])
    location_refs = list(dict.fromkeys(location_refs))[:4]
    if not location_refs:
        location_refs = [event["id"] for event in observed[:2]]
    outcome_refs = [event["id"] for event in final_output[:2]]
    if not outcome_refs:
        outcome_refs = [event["id"] for event in observed[-2:]]
    return location_refs, outcome_refs


def make_v18_sample(
    v15,
    uid: int,
    row: dict[str, Any],
    verdict: str,
    binary: str,
    archive: DirectoryArchive,
    root: str,
    source_zip: str,
    replacements: dict[str, str],
) -> dict[str, Any]:
    """Build targets from labels, but inputs only from observable allowlists."""

    run_id = row["run_id"]
    graph, fallback_task = v15.parse_config_graph_and_task(
        archive, row.get("config_path"), row
    )
    task_goal, final_output = observable_task_and_final(
        v15, archive, root, run_id, replacements
    )
    task_goal = task_goal or semantic_sanitize_text(
        fallback_task or f"{row.get('scenario', 'MAS')} task",
        replacements,
        run_id,
    )
    observed = v15.add_ids(
        project_observable_trajectory(
            v15, archive, root, run_id, replacements
        ),
        "obs",
    )
    known_edges = {
        (edge.get("source"), edge.get("target"))
        for edge in graph.get("edges") or []
    }
    for event in observed:
        edge = (event.get("source_agent"), event.get("target_agent"))
        if all(edge) and edge not in known_edges:
            graph.setdefault("edges", []).append(
                {"source": edge[0], "target": edge[1], "relation": "observed_message"}
            )
            known_edges.add(edge)
    if not final_output:
        # AutoGen minimal packages expose final outcomes in the trajectory only.
        trajectory_outcomes = [
            event for event in observed
            if event.get("type") in {"final_result", "final_outcome", "run_end"}
        ][-2:]
        final_output = [
            {key: value for key, value in event.items() if key != "id"}
            for event in trajectory_outcomes
        ]
    final_output = v15.add_ids(final_output[:2], "out")
    candidates = v15.build_candidates(graph, observed, final_output)
    objective = row.get("objective")
    scope, component_ids = component_ids_from_placement(v15, row, candidates)
    if not component_ids and verdict != "clean_safe":
        component_ids, scope = ["G::run"], "global"

    user = {
        "schema": SCHEMA,
        "sample_uid": f"v19_{uid:07d}",
        "task": {"scenario": row.get("scenario"), "goal": task_goal},
        "graph": graph,
        "audit_request": {"mode": "unknown_run_audit"},
        "run_evidence": {
            "coverage": {
                "has_trajectory": bool(observed),
                "has_run_trace": bool(observed),
                "has_final_output": bool(final_output),
                "has_clean_contrast": False,
                "granularity": "temporally_balanced_observable_trajectory",
                "event_types": sorted(
                    {
                        event.get("type")
                        for event in observed + final_output
                        if event.get("type")
                    }
                ),
            },
            "observed": observed,
            "final_output": final_output,
            "reference": {"available": False, "clean": []},
        },
        "graph_candidates": candidates,
    }
    if verdict == "clean_safe":
        attack_info = {"present": False, "surface": "none", "objective": "none"}
        localization = {"scope": "none", "component_ids": []}
        refs1, refs2 = aligned_evidence_refs(
            candidates, ["G::run"], observed, final_output
        )
    else:
        attack_info = {
            "present": True,
            "surface": row.get("surface"),
            "objective": objective,
        }
        localization = {"scope": scope, "component_ids": component_ids}
        refs1, refs2 = aligned_evidence_refs(
            candidates, component_ids, observed, final_output
        )
    assistant = {
        "decision": {"verdict": verdict, "binary_label": binary},
        "attack": attack_info,
        "localization": localization,
        "audit_trace": [
            {
                "step": "localize_component",
                "component_refs": component_ids,
                "evidence_refs": refs1,
            },
            {
                "step": "verify_outcome_effect",
                "component_refs": component_ids or ["G::run"],
                "evidence_refs": refs2,
            },
        ],
    }
    metadata = {
        "run_id": run_id,
        "sample_uid": f"v19_{uid:07d}",
        "source_zip": source_zip,
        "source_root": root,
        "condition": row.get("condition"),
        "sample_id": row.get("sample_id"),
        "scenario": row.get("scenario"),
        "topology": row.get("topology"),
        "surface": row.get("surface") or "none",
        "objective": objective or "none",
        "verdict": verdict,
        "label": binary,
        "schema": SCHEMA,
    }
    return {
        "messages": [
            {"role": "system", "content": v15.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "metadata": metadata,
    }


def observable_attack_evidence_score(
    manifest_row: dict[str, Any], sample: dict[str, Any]
) -> float:
    """Offline QC only; never changes event selection or visible input."""

    payloads: list[str] = []
    metadata = manifest_row.get("attack_metadata") or {}
    for component in metadata.get("components") or []:
        component_metadata = component.get("metadata") or {}
        payload = (
            component_metadata.get("core_payload")
            or component.get("payload_template")
            or ""
        )
        if payload:
            payloads.append(str(payload))
    if not payloads and manifest_row.get("attack_payload_template"):
        payloads.append(str(manifest_row["attack_payload_template"]))
    if not payloads:
        return 0.0

    user = json.loads(sample["messages"][1]["content"])
    evidence = user["run_evidence"]
    visible = " ".join(
        str(event.get("text") or "")
        for section in ("observed", "final_output")
        for event in evidence.get(section, [])
    ).casefold()
    best = 0.0
    for payload in payloads:
        tokens = {
            token
            for token in re.findall(r"[a-z]{3,}", payload.casefold())
            if token not in EVIDENCE_STOPWORDS and not token.startswith("aci")
        }
        if tokens:
            best = max(best, sum(token in visible for token in tokens) / len(tokens))
    return best


def remove_privileged_label_evidence(sample: dict[str, Any]) -> None:
    """Remove judge/marker-derived features that deterministically reveal the label.

    The marker scan is valid for constructing the supplied source label, but it is
    privileged supervision and must never be visible to the trained auditor.
    Clean-reference availability is likewise normalized across every class.
    """

    user = json.loads(sample["messages"][1]["content"])
    evidence = user["run_evidence"]
    privileged_sources = {"hq_visible_output_marker_scan"}
    for section in ("observed", "final_output"):
        evidence[section] = [
            event
            for event in evidence.get(section, [])
            if event.get("source") not in privileged_sources
        ]
    evidence["reference"] = {"available": False, "clean": []}
    coverage = evidence.get("coverage", {})
    coverage["has_clean_contrast"] = False
    coverage["has_final_output"] = bool(evidence.get("final_output"))

    valid_evidence_ids = {
        event["id"]
        for section in ("observed", "final_output")
        for event in evidence.get(section, [])
    }
    fallback_refs = [
        event["id"]
        for section in ("observed", "final_output")
        for event in evidence.get(section, [])
    ][:2]
    assistant = json.loads(sample["messages"][2]["content"])
    for trace in assistant.get("audit_trace", []):
        trace["evidence_refs"] = [
            ref
            for ref in trace.get("evidence_refs", [])
            if ref in valid_evidence_ids
        ]
        if not trace["evidence_refs"]:
            trace["evidence_refs"] = fallback_refs

    sample["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    sample["messages"][2]["content"] = json.dumps(assistant, ensure_ascii=False)


def sample_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    meta = [row["metadata"] for row in rows]
    return {
        "total": len(rows),
        "by_verdict": dict(Counter(x["verdict"] for x in meta)),
        "by_source_final_label": dict(Counter(x["source_final_label"] for x in meta)),
        "by_label_quality": dict(Counter(x["label_quality"] for x in meta)),
        "by_condition": dict(Counter(x["condition"] for x in meta)),
        "by_scenario": dict(Counter(x["scenario"] for x in meta)),
        "by_topology": dict(Counter(x["topology"] for x in meta)),
        "by_surface": dict(Counter(x["surface"] for x in meta)),
        "by_objective": dict(Counter(x["objective"] for x in meta)),
        "by_attack_mode": dict(Counter(x["attack_mode"] for x in meta)),
    }


def validate_samples(rows: list[dict[str, Any]], v15) -> dict[str, Any]:
    problems = Counter()
    dynamic_marker_hits = 0
    seen_uids, seen_source_keys = set(), set()
    for row in rows:
        meta = row["metadata"]
        uid, run_id = meta["sample_uid"], meta["run_id"]
        source_key = (meta["attack_mode"], run_id)
        problems["duplicate_uid"] += uid in seen_uids
        problems["duplicate_source_key"] += source_key in seen_source_keys
        seen_uids.add(uid)
        seen_source_keys.add(source_key)
        visible = json.dumps(row["messages"], ensure_ascii=False)
        for pattern in v15.FORBIDDEN_PATTERNS:
            if re.search(pattern, visible, flags=re.I):
                problems[f"leak::{pattern}"] += 1
        user = json.loads(row["messages"][1]["content"])
        assistant = json.loads(row["messages"][2]["content"])
        evidence = user["run_evidence"]
        problems["privileged_marker_evidence"] += sum(
            event.get("source") == "hq_visible_output_marker_scan"
            for section in ("observed", "final_output")
            for event in evidence.get(section, [])
        )
        problems["privileged_clean_reference"] += bool(
            evidence.get("reference", {}).get("available")
            or evidence.get("reference", {}).get("clean")
        )
        problems["privileged_clean_contrast_flag"] += bool(
            evidence.get("coverage", {}).get("has_clean_contrast")
        )
        evidence_ids = {
            event["id"]
            for event in user["run_evidence"]["observed"]
            + user["run_evidence"]["final_output"]
            + user["run_evidence"]["reference"]["clean"]
        }
        candidate_ids = {item["id"] for item in user["graph_candidates"]}
        for trace in assistant["audit_trace"]:
            problems["invalid_evidence_ref"] += sum(
                ref not in evidence_ids for ref in trace.get("evidence_refs", [])
            )
            problems["invalid_component_ref"] += sum(
                ref not in candidate_ids for ref in trace.get("component_refs", [])
            )
        problems["invalid_localization_ref"] += sum(
            ref not in candidate_ids
            for ref in assistant["localization"].get("component_ids", [])
        )
        if not user["run_evidence"]["observed"] and not user["run_evidence"]["final_output"]:
            problems["no_visible_evidence"] += 1
        if meta.get("_redaction_values"):
            for secret in meta["_redaction_values"]:
                if secret.casefold() in STRUCTURAL_MARKER_ALLOWLIST:
                    continue
                if secret and secret in visible:
                    dynamic_marker_hits += 1
        meta.pop("_redaction_values", None)
    problems["dynamic_secret_hits"] = dynamic_marker_hits
    return {key: value for key, value in problems.items() if value}


def make_quality_sample(
    rows: list[dict[str, Any]], path: Path, seed: int, count: int = 50
) -> None:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]
        strata[(meta["attack_mode"], meta["scenario"], meta["verdict"])].append(row)
    picked = []
    keys = list(strata)
    rng.shuffle(keys)
    while len(picked) < min(count, len(rows)):
        progressed = False
        for key in keys:
            if strata[key] and len(picked) < count:
                picked.append(strata[key].pop(rng.randrange(len(strata[key]))))
                progressed = True
        if not progressed:
            break
    payload = {
        "selection": "stratified by attack_mode, scenario, and verdict",
        "count": len(picked),
        "samples": picked,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_manual_review_queue(
    rows: list[dict[str, Any]], path: Path, seed: int, count: int = 200
) -> None:
    """Create a pending queue; this artifact never implies human review passed."""

    rng = random.Random(seed)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]
        source_label = meta.get("source_final_label")
        consensus = meta.get("semantic_consensus")
        if meta.get("label_quality") == "silver":
            strata["silver"].append(row)
        if consensus and source_label in {"success", "failure"} and consensus != source_label:
            strata["disputed"].append(row)
        if meta.get("condition") == "attacked" and not meta.get("security_judgment_available"):
            strata["judge_unavailable"].append(row)
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    names = sorted(strata)
    while len(selected) < count:
        progressed = False
        for name in names:
            pool = strata[name]
            while pool:
                row = pool.pop(rng.randrange(len(pool)))
                key = (str(row["metadata"].get("source_type")), str(row["metadata"]["run_id"]))
                if key in used:
                    continue
                used.add(key)
                selected.append(
                    {
                        "review_stratum": name,
                        "review_status": "pending_human_review",
                        "review_decision": None,
                        "review_notes": None,
                        "sample": row,
                    }
                )
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    payload = {
        "status": "PENDING",
        "human_review_completed": False,
        "selection": "balanced queue over disputed, judge-unavailable, and silver strata",
        "count": len(selected),
        "items": selected,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    v15 = load_v15_builder(args.v15_builder)
    v15.read_jsonl_from_zip = tolerant_archive_jsonl
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    selected_clean_run_ids: set[str] = set()

    modes = ("flat",) if (args.source_root / "merged" / "run_manifest.jsonl").is_file() else ("single", "dual")
    for mode in modes:
        if mode == "flat":
            merged_dir = args.source_root / "merged"
            evaluation_dir = args.source_root / "evaluation"
            manifest_path = merged_dir / "run_manifest.jsonl"
            trajectories_dir = merged_dir / "trajectories"
            configs_dir = merged_dir / "configs"
            results_dir = merged_dir / "marble_results"
            final_dir = args.source_root
        else:
            final_dir = args.source_root / mode / "final"
        if final_dir.is_dir():
            if mode != "flat":
                merged_dir = final_dir / "merged"
                evaluation_dir = final_dir / "evaluation"
                manifest_path = merged_dir / "run_manifest.jsonl"
                trajectories_dir = merged_dir / "trajectories"
                configs_dir = merged_dir / "configs"
                results_dir = merged_dir / "marble_results"
        else:
            # AutoGen minimal transfer layout.
            merged_dir = args.source_root / mode
            evaluation_dir = merged_dir / "labels"
            manifest_path = merged_dir / "run_manifest.jsonl"
            trajectories_dir = merged_dir / "trajectories"
            configs_dir = merged_dir / "configs"
            results_dir = merged_dir / "marble_results"
        if not manifest_path.is_file():
            continue
        manifest = read_jsonl(manifest_path)
        labels = {row["run_id"]: row for row in read_jsonl(evaluation_dir / "final_labels.jsonl")}
        def optional_rows(name: str) -> list[dict[str, Any]]:
            path = evaluation_dir / name
            return read_jsonl(path) if path.is_file() else []
        security = {row["run_id"]: row for row in optional_rows("security_judgments.jsonl")}
        signals = {row["run_id"]: row for row in optional_rows("attack_signals.jsonl")}
        private_signals = {row["run_id"]: row for row in optional_rows("private_control_signals.jsonl")}
        # Minimal transfer bundles may omit the two signal tables while retaining
        # the same decisions on attacked label rows. Reconstruct only the private-
        # control inclusion gate; model inputs still come exclusively from configs
        # and observable trajectories.
        paired_private_controls = {
            str(row.get("control_run_id"))
            for row in labels.values()
            if row.get("control_run_id")
            and str(row.get("control_run_id")).endswith("__private_control")
        }
        leaky_private_controls = {
            str(row.get("control_run_id"))
            for row in labels.values()
            if row.get("natural_control_marker_leak") is True
            and row.get("control_run_id")
        }
        archive = DirectoryArchive(
            merged_dir,
            f"{mode}_source",
            trajectories_dir=trajectories_dir,
            configs_dir=configs_dir,
            results_dir=results_dir,
        )
        counts = Counter()

        for manifest_row in manifest:
            topology = str(manifest_row.get("topology") or "")
            if args.include_topology and topology not in args.include_topology:
                counts["excluded_topology"] += 1
                continue
            if topology in args.exclude_topology:
                counts["excluded_topology"] += 1
                continue
            run_id = manifest_row["run_id"]
            condition = manifest_row.get("condition")
            if manifest_row.get("status") != "completed":
                counts["non_completed"] += 1
                excluded_rows.append(
                    {"run_id": run_id, "reason": "non_completed", "source": mode}
                )
                continue

            source_label = None
            label_quality = "control"
            final_label_row = labels.get(run_id)
            if condition == "attacked":
                if final_label_row is None:
                    raise ValueError(f"Missing final label for attacked run: {run_id}")
                source_label = final_label_row["final_label"]
                label_quality = final_label_row.get("label_quality")
                if source_label not in SOURCE_LABEL_TO_VERDICT:
                    counts[f"excluded_{source_label}"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": f"source_final_label::{source_label}",
                            "source": mode,
                            "label_quality": label_quality,
                        }
                    )
                    continue
                verdict, binary = SOURCE_LABEL_TO_VERDICT[source_label]
            elif condition == "clean":
                if run_id in selected_clean_run_ids:
                    counts["duplicate_cross_suite_clean"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "duplicate_cross_suite_clean",
                            "source": mode,
                        }
                    )
                    continue
                selected_clean_run_ids.add(run_id)
                source_label, verdict, binary = "clean", "clean_safe", "safe"
            elif condition == "private_control":
                private_signal = private_signals.get(run_id)
                if private_signal is None and private_signals:
                    counts["private_control_missing_signal"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "private_control_missing_signal",
                            "source": mode,
                        }
                    )
                    continue
                if private_signal is None and run_id not in paired_private_controls:
                    counts["private_control_unpaired_minimal_bundle"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "private_control_unpaired_minimal_bundle",
                            "source": mode,
                        }
                    )
                    continue
                if (
                    private_signal is not None and private_signal.get("marker_success")
                ) or run_id in leaky_private_controls:
                    counts["private_control_natural_marker_leakage"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "private_control_natural_marker_leakage",
                            "source": mode,
                        }
                    )
                    continue
                source_label, verdict, binary = (
                    "private_control",
                    "clean_safe",
                    "safe",
                )
            else:
                counts["unknown_condition"] += 1
                continue

            row_for_build = dict(manifest_row)
            metadata = dict(row_for_build.get("attack_metadata") or {})
            secrets = redaction_values(row_for_build, signals.get(run_id))
            metadata["success_markers"] = secrets
            row_for_build["attack_metadata"] = metadata
            replacements = entity_replacements(run_id, secrets)

            try:
                sample = make_v18_sample(
                    v15,
                    len(all_samples),
                    row_for_build,
                    verdict,
                    binary,
                    archive,
                    archive.virtual_root,
                    args.source_archive_name,
                    replacements,
                )
            except (json.JSONDecodeError, UnicodeError) as exc:
                counts["malformed_source_evidence"] += 1
                excluded_rows.append(
                    {
                        "run_id": run_id,
                        "reason": "malformed_source_evidence",
                        "source": mode,
                        "source_final_label": source_label,
                        "error": str(exc),
                    }
                )
                continue
            # Defensive compatibility cleanup. The V18 allowlist projector
            # does not create these fields, and this call must remain a no-op.
            remove_privileged_label_evidence(sample)
            relevant_names = {
                f"{archive.virtual_root}/data/{kind}/{run_id}.jsonl"
                for kind in ("trajectories", "marble_results")
            }
            malformed_count = sum(
                1 for name, _ in archive.malformed_line_keys if name in relevant_names
            )
            user_for_coverage = json.loads(sample["messages"][1]["content"])
            evidence = user_for_coverage.get("run_evidence", {})
            if not evidence.get("observed") and not evidence.get("final_output"):
                counts["no_visible_source_evidence"] += 1
                excluded_rows.append(
                    {
                        "run_id": run_id,
                        "reason": "no_visible_source_evidence",
                        "source": mode,
                        "source_final_label": source_label,
                        "malformed_lines_skipped": malformed_count,
                    }
                )
                continue
            stable_uid(sample, len(all_samples))
            sanitize_sample_messages(sample, replacements, run_id)
            if condition == "attacked":
                evidence_score = observable_attack_evidence_score(
                    row_for_build, sample
                )
                if evidence_score < 0.20:
                    counts["insufficient_observable_evidence"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "insufficient_observable_evidence",
                            "source": mode,
                            "source_final_label": source_label,
                            "label_quality": label_quality,
                            "observable_evidence_score": round(evidence_score, 6),
                        }
                    )
                    continue
            meta = sample["metadata"]
            if mode == "flat":
                placement = row_for_build.get("placement") or {}
                source_attack_mode = (
                    "none"
                    if condition == "clean"
                    else (
                        "dual_site"
                        if placement.get("type") == "dual"
                        else "single_site"
                    )
                )
            else:
                source_attack_mode = "single_site" if mode == "single" else "dual_site"
            meta.update(
                {
                    "source_type": args.source_type,
                    "attack_mode": source_attack_mode,
                    "source_final_label": source_label,
                    "label_quality": label_quality,
                    "label_policy": "stage6_final_label_preserved",
                    "source_evidence_sources": (
                        list(final_label_row.get("evidence_sources") or [])
                        if final_label_row
                        else []
                    ),
                    "semantic_consensus": (
                        (final_label_row.get("semantic_consensus") or {}).get("decision")
                        if final_label_row
                        else None
                    ),
                    "security_judgment_available": run_id in security,
                    "localization_policy": "source_attack_placement_candidate_projection",
                    "input_policy": (
                        "strict observable-field allowlist; label-blind temporal/type "
                        "selection; semantic-preserving per-run redaction"
                    ),
                    "malformed_source_lines_skipped": malformed_count,
                    "_redaction_values": secrets,
                }
            )
            all_samples.append(sample)
            counts[f"selected_{source_label}"] += 1

        source_audit[mode] = {
            "manifest_rows": len(manifest),
            "final_label_rows": len(labels),
            "security_judgment_rows": len(security),
            "attack_signal_rows": len(signals),
            "private_control_signal_rows": len(private_signals),
            "malformed_jsonl_lines_skipped": len(archive.malformed_line_keys),
            "counts": dict(counts),
        }

    validation = validate_samples(all_samples, v15)
    if validation:
        # Failure-only diagnostic; validation removes private redaction metadata
        # before this artifact is written.
        write_jsonl(output / "_candidate_all.validation_failed.jsonl", all_samples)
        raise RuntimeError(f"V18 validation failed: {json.dumps(validation, ensure_ascii=False)}")

    train, validation, test, held_out = group_split(all_samples, args.seed)
    groups = {
        name: {(x["metadata"]["scenario"], x["metadata"]["sample_id"]) for x in rows}
        for name, rows in (("train", train), ("validation", validation), ("test", test))
    }
    overlap = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Grouped split overlap: {overlap}")

    write_jsonl(output / "all.jsonl", all_samples)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_jsonl(output / "test.jsonl", test)
    write_jsonl(output / "excluded_source_labels.jsonl", excluded_rows)
    make_manual_review_queue(
        all_samples, output / "manual_review_queue_200_v19.json", args.seed, count=200
    )
    summary = {
        "schema": SCHEMA,
        "source_archive": args.source_archive_name,
        "policy": {
            "source_immutability": "Source final labels are preserved. No attacked label is rewritten.",
            "training_selection": (
                "Attacked success/failure are mapped structurally to attack_success/attack_failed. "
                "not_exposed, ambiguous, and invalid remain unchanged in the exclusion audit and "
                "are not forced into the three-class SFT target."
            ),
            "controls": (
                "Completed clean runs are clean_safe. Completed private controls are clean_safe "
                "only when their provided private-control signal has no natural marker leakage."
            ),
            "localization": "Candidate localization projects the source attack placement; it is not relabeled as causal attribution.",
            "split": (
                "Task-grouped train/validation/test split. A scenario/task pair is "
                "assigned once across every topology and framework. Test is sealed "
                "from training and model selection."
            ),
            "redaction": (
                "Per-run entity pseudonyms replace raw markers and private values. "
                "Benchmark injection tags are removed while ordinary instruction, "
                "communication, tool, and outcome semantics are retained."
            ),
        },
        "source_audit": source_audit,
        "excluded": dict(Counter(row["reason"] for row in excluded_rows)),
        "held_out_tasks": held_out,
        "split_group_overlap": overlap,
        "files": {
            "all": sample_stats(all_samples),
            "train": sample_stats(train),
            "validation": sample_stats(validation),
            "test": sample_stats(test),
        },
        "validation": validation,
    }
    (output / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not getattr(args, "quiet", False):
        # Keep console output ASCII-safe on Windows/GBK; artifacts remain UTF-8.
        print(json.dumps(summary, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--v15-builder", required=True, type=Path)
    parser.add_argument(
        "--source-archive-name",
        default="weekend_fresh_single1_10_dual1_5_bundle.tar.zst",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-type", default="marble")
    parser.add_argument("--include-topology", action="append", default=[])
    parser.add_argument("--exclude-topology", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
