from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


TRACKS = {
    "marble_mab": {"train": 5404, "validation": 1756, "sealed_test": 1522},
    "autogen_mab": {"train": 1912, "validation": 792, "sealed_test": 624},
    "marble_appworld": {"train": 3122, "validation": 406, "sealed_test": 393},
}
VERDICTS = ("clean_safe", "attack_failed", "attack_success")
SCOPES = {"none", "global", "node", "edge", "tool", "multi"}
SURFACES = {"none", "input", "message", "tool", "memory", "profile", "dual"}
OBJECTIVES = {"none", "disclosure", "disruption", "hijacking"}
PRIVILEGED_KEYS = {
    "attack_exposed", "attack_injected", "attack_active", "attack_payload_template",
    "source_final_label", "marker_success", "success_markers", "security_judgment",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path: Path) -> list[tuple[str, dict]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append((line.rstrip("\r\n"), json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
    return result


def nested_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def nonempty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_row(row: dict, track: str, split: str) -> list[str]:
    errors = []
    run_id = str(row.get("metadata", {}).get("run_id") or "")
    prefix = f"{track}/{split}/{run_id or '<missing>'}"
    messages = row.get("messages", [])
    if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        return [f"{prefix}: bad message role contract"]
    try:
        user = json.loads(messages[1]["content"])
        report = json.loads(messages[2]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return [f"{prefix}: unparseable user/assistant JSON: {exc}"]
    leaked = sorted(PRIVILEGED_KEYS & nested_keys(user))
    if leaked:
        errors.append(f"{prefix}: privileged user keys: {leaked}")
    if "audit_control" in user:
        errors.append(f"{prefix}: base data unexpectedly contains audit_control")
    if set(report) != {"decision", "attack", "localization", "audit_trace"}:
        errors.append(f"{prefix}: unexpected base report top-level schema")
        return errors
    verdict = report.get("decision", {}).get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"{prefix}: invalid verdict {verdict!r}")
    if row.get("metadata", {}).get("verdict") != verdict:
        errors.append(f"{prefix}: metadata/report verdict mismatch")
    expected_binary = "unsafe" if verdict == "attack_success" else "safe"
    if report.get("decision", {}).get("binary_label") != expected_binary:
        errors.append(f"{prefix}: verdict/binary-label semantic mismatch")
    attack = report.get("attack", {})
    expected_present = verdict != "clean_safe"
    if attack.get("present") is not expected_present:
        errors.append(f"{prefix}: verdict/attack-present semantic mismatch")
    if attack.get("surface") not in SURFACES or attack.get("objective") not in OBJECTIVES:
        errors.append(f"{prefix}: invalid attack surface/objective taxonomy")
    if verdict == "clean_safe" and (attack.get("surface"), attack.get("objective")) != ("none", "none"):
        errors.append(f"{prefix}: clean row has non-none attack characterization")
    if verdict != "clean_safe" and (attack.get("surface") == "none" or attack.get("objective") == "none"):
        errors.append(f"{prefix}: attack row lacks attack characterization")
    evidence_rows = (
        user.get("run_evidence", {}).get("observed", [])
        + user.get("run_evidence", {}).get("final_output", [])
    )
    evidence_ids = [str(item.get("id")) for item in evidence_rows if isinstance(item, dict) and item.get("id")]
    if len(evidence_ids) != len(set(evidence_ids)) or not evidence_ids:
        errors.append(f"{prefix}: missing or duplicate evidence IDs")
    for item in evidence_rows:
        if not isinstance(item, dict) or not nonempty_text(item.get("text")):
            errors.append(f"{prefix}: empty or malformed observable evidence text")
            break
    candidate_ids = [str(item.get("id")) for item in user.get("graph_candidates", []) if isinstance(item, dict)]
    if len(candidate_ids) != len(set(candidate_ids)) or not candidate_ids:
        errors.append(f"{prefix}: missing or duplicate graph candidate IDs")
    for candidate in user.get("graph_candidates", []):
        if not isinstance(candidate, dict) or not nonempty_text(candidate.get("id")) or not nonempty_text(candidate.get("type")):
            errors.append(f"{prefix}: malformed graph candidate identity")
            break
        if "description" in candidate and not nonempty_text(candidate.get("description")):
            errors.append(f"{prefix}: empty graph candidate description")
            break
        candidate_refs = {
            str(value) for key, values in candidate.items() if key.endswith("event_refs") and isinstance(values, list)
            for value in values
        }
        if candidate_refs and not candidate_refs.issubset(set(evidence_ids)):
            errors.append(f"{prefix}: graph candidate references invisible evidence")
            break
    components = [str(value) for value in report.get("localization", {}).get("component_ids", [])]
    if not set(components).issubset(set(candidate_ids)):
        errors.append(f"{prefix}: localization references invisible candidates")
    scope = report.get("localization", {}).get("scope")
    if scope not in SCOPES:
        errors.append(f"{prefix}: invalid localization scope")
    if verdict == "clean_safe" and (scope != "none" or components):
        errors.append(f"{prefix}: clean row has non-empty localization")
    if verdict != "clean_safe" and (scope == "none" or not components):
        errors.append(f"{prefix}: attack row lacks localization")
    traces = report.get("audit_trace", [])
    if not isinstance(traces, list) or not traces:
        errors.append(f"{prefix}: empty audit_trace")
    else:
        for trace in traces:
            if trace.get("step") not in {"localize_component", "verify_outcome_effect"}:
                errors.append(f"{prefix}: invalid audit trace step")
                break
            refs = [str(value) for value in trace.get("evidence_refs", [])]
            if not refs or not set(refs).issubset(set(evidence_ids)):
                errors.append(f"{prefix}: invalid or empty audit evidence_refs")
                break
            component_refs = [str(value) for value in trace.get("component_refs", [])]
            if not set(component_refs).issubset(set(candidate_ids)):
                errors.append(f"{prefix}: audit trace references invisible components")
                break
        if {trace.get("step") for trace in traces} != {"localize_component", "verify_outcome_effect"}:
            errors.append(f"{prefix}: incomplete audit trace semantics")
    return errors


def task_key(row: dict, track: str) -> tuple[str, str, str]:
    meta = row["metadata"]
    family = "appworld" if track == "marble_appworld" else "multiagentbench"
    return family, str(meta.get("scenario")), str(meta.get("sample_id"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe three-track V22-ALL source bundle.")
    parser.add_argument("--marble-mab-data", required=True, type=Path)
    parser.add_argument("--autogen-mab-data", required=True, type=Path)
    parser.add_argument("--marble-appworld-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-per-verdict", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    inputs = {
        "marble_mab": args.marble_mab_data.resolve(),
        "autogen_mab": args.autogen_mab_data.resolve(),
        "marble_appworld": args.marble_appworld_data.resolve(),
    }
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    loaded: dict[str, dict[str, list[tuple[str, dict]]]] = defaultdict(dict)
    manifest_tracks = {}
    try:
        for track, root in inputs.items():
            if not root.is_dir():
                problems.append(f"{track}: missing data directory {root}")
                continue
            info = {"source_dir": str(root), "splits": {}, "sealed_test_rows": TRACKS[track]["sealed_test"]}
            for split in ("train", "validation"):
                path = root / f"{split}.jsonl"
                if not path.is_file():
                    problems.append(f"{track}: missing {path}")
                    continue
                loaded[track][split] = read_lines(path)
                expected = TRACKS[track][split]
                if len(loaded[track][split]) != expected:
                    problems.append(f"{track}/{split}: expected {expected}, got {len(loaded[track][split])}")
                info["splits"][split] = {"rows": len(loaded[track][split]), "sha256": sha256(path)}
            manifest_tracks[track] = info

        all_run_ids: dict[str, tuple[str, str]] = {}
        split_run_ids = {"train": set(), "validation": set()}
        split_inputs = {"train": set(), "validation": set()}
        split_tasks = {"train": set(), "validation": set()}
        sample_cells = {}
        rng = random.Random(args.seed)
        for track in TRACKS:
            for split in ("train", "validation"):
                cells: dict[str, list[tuple[str, dict]]] = defaultdict(list)
                for line, row in loaded.get(track, {}).get(split, []):
                    problems.extend(validate_row(row, track, split))
                    run_id = str(row.get("metadata", {}).get("run_id") or "")
                    if not run_id:
                        problems.append(f"{track}/{split}: missing run_id")
                        continue
                    if run_id in all_run_ids:
                        problems.append(f"duplicate run_id {run_id}: {all_run_ids[run_id]} and {(track, split)}")
                    all_run_ids[run_id] = (track, split)
                    split_run_ids[split].add(run_id)
                    split_inputs[split].add(sha256_bytes(row["messages"][1]["content"].encode("utf-8")))
                    split_tasks[split].add(task_key(row, track))
                    verdict = str(row.get("metadata", {}).get("verdict"))
                    cells[verdict].append((line, row))
                for verdict in VERDICTS:
                    pool = cells.get(verdict, [])
                    cell = f"{track}/{split}/{verdict}"
                    if len(pool) < args.sample_per_verdict:
                        problems.append(f"{cell}: only {len(pool)} rows; need {args.sample_per_verdict}")
                        continue
                    chosen = rng.sample(pool, args.sample_per_verdict)
                    sample_problems = []
                    for _, sampled_row in chosen:
                        sample_problems.extend(validate_row(sampled_row, track, split))
                    if sample_problems:
                        problems.extend(f"sample-semantic-gate: {value}" for value in sample_problems)
                    sample_path = output / "quality_samples" / track / split / f"{verdict}.jsonl"
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    sample_path.write_text("\n".join(line for line, _ in chosen) + "\n", encoding="utf-8")
                    sample_cells[cell] = {
                        "available": len(pool), "sampled": len(chosen), "sha256": sha256(sample_path),
                        "semantic_contract_checks": "PASS",
                    }

        overlaps = {
            "run_id": len(split_run_ids["train"] & split_run_ids["validation"]),
            "exact_user_input": len(split_inputs["train"] & split_inputs["validation"]),
            "task_group": len(split_tasks["train"] & split_tasks["validation"]),
        }
        for name, count in overlaps.items():
            if count:
                problems.append(f"train/validation {name} overlap: {count}")

        quality = {
            "version": "V22-ALL-preupload-quality-v2",
            "status": "PASS" if not problems else "FAIL",
            "seed": args.seed,
            "sample_per_track_split_verdict": args.sample_per_verdict,
            "sampled_rows": sum(cell["sampled"] for cell in sample_cells.values()),
            "sample_cells": sample_cells,
            "full_dataset_checks": {
                "rows_structurally_checked": sum(
                    len(loaded.get(track, {}).get(split, [])) for track in TRACKS for split in ("train", "validation")
                ),
                "train_validation_overlap": overlaps,
                "privileged_input_keys": 0 if not any("privileged" in value for value in problems) else None,
                "invalid_evidence_or_components": 0 if not any("evidence" in value or "candidate" in value for value in problems) else None,
                "semantic_consistency_failures": 0 if not any("semantic" in value or "characterization" in value or "localization" in value for value in problems) else None,
                "sealed_test_accessed": False,
            },
            "problems": problems[:1000],
        }
        write_json(output / ("PREUPLOAD_QUALITY_GATE.json" if not problems else "QUALITY_FAILURE_REPORT.json"), quality)
        if problems:
            raise RuntimeError(f"V22-ALL pre-upload quality gate failed with {len(problems)} problem(s)")

        base = output / "base_dataset"
        index = output / "track_index"
        base.mkdir()
        index.mkdir()
        combined = {}
        for split in ("train", "validation"):
            combined_path = base / f"{split}.jsonl"
            track_index_path = index / f"{split}.jsonl"
            with combined_path.open("w", encoding="utf-8", newline="\n") as data_out, track_index_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as index_out:
                count = 0
                for track in TRACKS:
                    for line, row in loaded[track][split]:
                        data_out.write(line + "\n")
                        index_out.write(json.dumps({
                            "run_id": row["metadata"]["run_id"], "track": track,
                            "split": split, "verdict": row["metadata"]["verdict"],
                        }, ensure_ascii=False) + "\n")
                        count += 1
            combined[split] = {"rows": count, "sha256": sha256(combined_path), "index_sha256": sha256(track_index_path)}

        source_manifest = {
            "version": "V22-ALL-unified-source-v2",
            "tracks": manifest_tracks,
            "combined": combined,
            "classification_data": "unexpanded base_dataset",
            "audit_sft_data": "server-built after ModernBERT predictions and train-only teacher expansion",
            "preupload_quality_gate": "PASS",
            "quality_sample_rows": quality["sampled_rows"],
            "sealed_test_rows": sum(value["sealed_test"] for value in TRACKS.values()),
            "sealed_test_accessed": False,
        }
        write_json(output / "SOURCE_MANIFEST.json", source_manifest)
        archive = Path(shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name))
        print(json.dumps({"status": "PASS", "bundle": str(output), "archive": str(archive), **source_manifest}, ensure_ascii=False, indent=2))
    except Exception as exc:
        if not (output / "QUALITY_FAILURE_REPORT.json").exists():
            write_json(output / "QUALITY_FAILURE_REPORT.json", {
                "version": "V22-ALL-preupload-quality-v2", "status": "FAIL", "error": str(exc), "problems": problems,
            })
        print(json.dumps({"status": "FAIL", "report": str(output / "QUALITY_FAILURE_REPORT.json"), "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
