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


SCHEMA = "Graph-grounded-Candidate-SFT/v12"
STRUCTURAL_MARKER_ALLOWLIST = {"audit_trace"}
SOURCE_LABEL_TO_VERDICT = {
    "success": ("attack_success", "unsafe"),
    "failure": ("attack_failed", "safe"),
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
    spec = importlib.util.spec_from_file_location("v15_builder_for_v17", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V15 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectoryArchive:
    """Read-only adapter exposing a final/merged directory as the V15 virtual ZIP layout."""

    def __init__(self, merged_dir: Path, virtual_root: str):
        self.merged_dir = merged_dir
        self.virtual_root = virtual_root
        # A source file is read several times by the inherited V15 extraction
        # helpers. Track physical line locations so one malformed line is
        # counted once, rather than once per read.
        self.malformed_line_keys: set[tuple[str, int]] = set()
        self._names = [f"{virtual_root}/data/run_manifest.jsonl"]
        for kind in ("configs", "trajectories", "marble_results"):
            base = merged_dir / kind
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    """Hold out one task from 1-5 and one from 6-10 per scenario."""

    rng = random.Random(seed)
    scenarios = sorted({str(r["metadata"]["scenario"]) for r in rows})
    held_out: dict[str, list[int]] = {}
    for scenario in scenarios:
        ids = sorted(
            {
                int(r["metadata"]["sample_id"])
                for r in rows
                if r["metadata"]["scenario"] == scenario
                and r["metadata"].get("sample_id") is not None
            }
        )
        low = [x for x in ids if x <= 5]
        high = [x for x in ids if x > 5]
        chosen = []
        if low:
            chosen.append(rng.choice(low))
        if high:
            chosen.append(rng.choice(high))
        held_out[scenario] = sorted(chosen)

    train, test = [], []
    for row in rows:
        meta = row["metadata"]
        sid = int(meta["sample_id"])
        (test if sid in held_out[str(meta["scenario"])] else train).append(row)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test, held_out


def stable_uid(sample: dict[str, Any], uid: int) -> None:
    new_uid = f"v17_{uid:07d}"
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
    text = re.sub(r"\bBenchmark addendum\b", "Additional instruction", text, flags=re.I)
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
            message["content"] = json.dumps(structured, ensure_ascii=False)


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


def build(args: argparse.Namespace) -> None:
    v15 = load_v15_builder(args.v15_builder)
    v15.read_jsonl_from_zip = tolerant_archive_jsonl
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    selected_clean_run_ids: set[str] = set()

    for mode in ("single", "dual"):
        final_dir = args.source_root / mode / "final"
        merged_dir = final_dir / "merged"
        evaluation_dir = final_dir / "evaluation"
        manifest = read_jsonl(merged_dir / "run_manifest.jsonl")
        labels = {row["run_id"]: row for row in read_jsonl(evaluation_dir / "final_labels.jsonl")}
        security = {
            row["run_id"]: row
            for row in read_jsonl(evaluation_dir / "security_judgments.jsonl")
        }
        signals = {
            row["run_id"]: row
            for row in read_jsonl(evaluation_dir / "attack_signals.jsonl")
        }
        private_signals = {
            row["run_id"]: row
            for row in read_jsonl(evaluation_dir / "private_control_signals.jsonl")
        }
        archive = DirectoryArchive(merged_dir, f"{mode}_source")
        counts = Counter()

        for manifest_row in manifest:
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
                if private_signal is None:
                    counts["private_control_missing_signal"] += 1
                    excluded_rows.append(
                        {
                            "run_id": run_id,
                            "reason": "private_control_missing_signal",
                            "source": mode,
                        }
                    )
                    continue
                if private_signal.get("marker_success"):
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

            # The inherited V15 extractor is reused for schema compatibility,
            # but V17 replaces its destructive sanitizer and clean-content
            # preference with semantics-preserving variants for this run.
            v15.sanitize_text = (
                lambda value, markers=None, objective=None, _replacements=replacements, _run_id=run_id:
                semantic_sanitize_text(value, _replacements, _run_id)
            )
            v15.events_from_trajectory = (
                lambda archive_arg, root_arg, run_id_arg, markers_arg, objective_arg, _v15=v15:
                semantic_events_from_trajectory(
                    _v15,
                    archive_arg,
                    root_arg,
                    run_id_arg,
                    markers_arg,
                    objective_arg,
                )
            )

            try:
                sample = v15.make_sample(
                    len(all_samples),
                    row_for_build,
                    verdict,
                    binary,
                    archive,
                    archive.virtual_root,
                    args.source_archive_name,
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
            meta = sample["metadata"]
            meta.update(
                {
                    "source_type": "weekend_fresh_single1_10_dual1_5",
                    "attack_mode": "single_site" if mode == "single" else "dual_site",
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
                        "semantic_preserving_redaction; benchmark identity and raw "
                        "entities removed; ordinary instruction and propagation semantics retained"
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
        raise RuntimeError(f"V17 validation failed: {json.dumps(validation, ensure_ascii=False)}")

    train, test, held_out = group_split(all_samples, args.seed)
    train_groups = {(x["metadata"]["scenario"], x["metadata"]["sample_id"]) for x in train}
    test_groups = {(x["metadata"]["scenario"], x["metadata"]["sample_id"]) for x in test}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise RuntimeError(f"Grouped split overlap: {overlap}")

    write_jsonl(output / "all.jsonl", all_samples)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)
    write_jsonl(output / "excluded_source_labels.jsonl", excluded_rows)
    make_quality_sample(
        all_samples, output / "manual_quality_sample_50_v17.json", args.seed
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
            "split": "Task-group holdout: one task from 1-5 and one from 6-10 per scenario.",
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
            "test": sample_stats(test),
        },
        "validation": validation,
    }
    (output / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    build(parser.parse_args())


if __name__ == "__main__":
    main()
