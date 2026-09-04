from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")
SPLITS = ("train", "validation", "test")
DIMS = ("verdict", "condition", "scenario", "topology", "surface", "objective", "attack_mode", "label_quality", "source_type")
TRAIN_LEAK_RE = re.compile(
    r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|success_marker|success_markers|"
    r"attack_metadata|attack_id|marker_check|\[Injected[^\]]*\]|offline verifier|"
    r"attack-success index|labeled as attack-success", re.I
)
PRIVILEGED_KEYS = {
    "attack_exposed", "attack_injected", "attack_active", "attack_payload_template",
    "source_final_label", "marker_success", "success_marker", "success_markers",
    "security_judgment", "attack_metadata", "attack_id", "gold_component_labels",
    "private_user_information", "judge_output", "judge_outputs",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values: list[int]) -> dict[str, float | int]:
    values = sorted(values)
    def q(p: float):
        if not values: return 0
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] if lo == hi else round(values[lo] * (hi - x) + values[hi] * (x - lo), 2)
    return {"min": values[0] if values else 0, "p50": q(.5), "p90": q(.9), "p95": q(.95), "p99": q(.99), "max": values[-1] if values else 0}


def nested_keys(value) -> set[str]:
    out = set()
    if isinstance(value, dict):
        for key, child in value.items():
            out.add(str(key))
            out.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value: out.update(nested_keys(child))
    return out


def shape(value):
    if isinstance(value, dict): return {k: shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list): return [shape(value[0])] if value else []
    return type(value).__name__


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(shape(value), sort_keys=True).encode()).hexdigest()[:16]


def pct(counter: Counter) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    return {str(k): {"count": v, "percent": round(100 * v / total, 3)} for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v22", required=True, type=Path)
    ap.add_argument("--v23", required=True, type=Path)
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument("--candidate-root", required=True, type=Path)
    ap.add_argument("--validator", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    a = ap.parse_args()
    old_m = json.loads((a.v22 / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    new_m = json.loads((a.v23 / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    combined_m = json.loads((a.combined / "COMBINED_MANIFEST.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("v22q", a.validator)
    mod = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)

    counters = {origin: {d: Counter() for d in DIMS} for origin in ("V22", "V23_increment", "V23_total")}
    systems = {origin: Counter() for origin in ("V22", "V23_increment")}
    user_shapes = {origin: Counter() for origin in ("V22", "V23_increment")}
    target_shapes = {origin: Counter() for origin in ("V22", "V23_increment")}
    metadata_keys = {origin: Counter() for origin in ("V22", "V23_increment")}
    lengths = {origin: defaultdict(list) for origin in ("V22", "V23_increment")}
    leak_hits = Counter(); leak_examples = []
    privileged_hits = Counter(); privileged_examples = []
    input_uid_mismatch = []
    dataset_version = Counter()
    validation_errors = []
    by_track_split_origin = Counter()
    redaction_placeholders = {origin: Counter() for origin in ("V22", "V23_increment")}

    for track in TRACKS:
        for split in SPLITS:
            old_n = old_m["tracks"][track][split]["rows"]
            with (a.v23 / track / f"{split}.jsonl").open(encoding="utf-8") as f:
                for pos, line in enumerate((x for x in f if x.strip()), 1):
                    row = json.loads(line); origin = "V22" if pos <= old_n else "V23_increment"
                    by_track_split_origin[(track, split, origin)] += 1
                    for d in DIMS: counters[origin][d][str(row["metadata"].get(d))] += 1
                    systems[origin][row["messages"][0]["content"]] += 1
                    user = json.loads(row["messages"][1]["content"])
                    target = json.loads(row["messages"][2]["content"])
                    user_shapes[origin][fingerprint(user)] += 1
                    target_shapes[origin][fingerprint(target)] += 1
                    metadata_keys[origin][tuple(sorted(row["metadata"]))] += 1
                    lengths[origin]["user_chars"].append(len(row["messages"][1]["content"]))
                    lengths[origin]["target_chars"].append(len(row["messages"][2]["content"]))
                    lengths[origin]["events"].append(len(user["run_evidence"]["observed"]) + len(user["run_evidence"]["final_output"]))
                    lengths[origin]["candidates"].append(len(user["graph_candidates"]))
                    visible = json.dumps(row["messages"][:2], ensure_ascii=False)
                    match = TRAIN_LEAK_RE.search(visible)
                    if match:
                        leak_hits[match.group(0).lower()] += 1
                        if len(leak_examples) < 20: leak_examples.append({"run_id": row["metadata"]["run_id"], "term": match.group(0)})
                    pkeys = PRIVILEGED_KEYS & nested_keys(user)
                    for key in pkeys: privileged_hits[key] += 1
                    if pkeys and len(privileged_examples) < 20: privileged_examples.append({"run_id": row["metadata"]["run_id"], "keys": sorted(pkeys)})
                    uid = row["metadata"].get("sample_uid")
                    if user.get("sample_uid") != uid and len(input_uid_mismatch) < 20:
                        input_uid_mismatch.append(row["metadata"]["run_id"])
                    dataset_version[str(row["metadata"].get("dataset_version", "absent"))] += 1
                    for token in re.findall(r"\[(?:REDACTED_ENTITY_[A-F0-9]{10}|REDACTED_[A-Z_]+|PRIVATE_CONTEXT_REDACTED|EXTERNAL_CONTEXT_REDACTED)\]", visible):
                        redaction_placeholders[origin][token.split("_")[0] + "_..."] += 1
                    problems = mod.validate_row(row, track, split)
                    if problems and len(validation_errors) < 100: validation_errors.extend(problems)
                    for d in DIMS: counters["V23_total"][d][str(row["metadata"].get(d))] += 1

    excluded = Counter(); source_runs = 0; candidates = 0
    for p in sorted(a.candidate_root.glob("*/stats.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        source_runs += sum(v.get("manifest_rows", 0) for v in j.get("source_audit", {}).values())
        candidates += j["files"]["all"]["total"]
        excluded.update(j.get("excluded", {}))

    schema_alignment = {
        "same_system_prompt_set": set(systems["V22"]) == set(systems["V23_increment"]),
        "v22_system_prompts": len(systems["V22"]), "increment_system_prompts": len(systems["V23_increment"]),
        "same_user_shape_set": set(user_shapes["V22"]) == set(user_shapes["V23_increment"]),
        "increment_user_shapes_subset_of_v22": set(user_shapes["V23_increment"]) <= set(user_shapes["V22"]),
        "v22_user_shapes": dict(user_shapes["V22"]), "increment_user_shapes": dict(user_shapes["V23_increment"]),
        "same_target_shape_set": set(target_shapes["V22"]) == set(target_shapes["V23_increment"]),
        "v22_target_shapes": dict(target_shapes["V22"]), "increment_target_shapes": dict(target_shapes["V23_increment"]),
        "metadata_key_variants_v22": len(metadata_keys["V22"]), "metadata_key_variants_increment": len(metadata_keys["V23_increment"]),
        "input_metadata_uid_mismatches": input_uid_mismatch,
    }
    errors = []
    if validation_errors: errors.append("schema_or_semantic_validation_errors")
    if leak_hits: errors.append("sft_visible_forbidden_pattern_hits")
    if privileged_hits: errors.append("privileged_user_key_hits")
    if input_uid_mismatch: errors.append("sample_uid_mismatch")
    if not schema_alignment["same_system_prompt_set"]: errors.append("system_prompt_drift")
    if not schema_alignment["increment_user_shapes_subset_of_v22"]: errors.append("incompatible_user_schema_drift")
    if not schema_alignment["same_target_shape_set"]: errors.append("target_schema_drift")
    report = {
        "version": "V23-comprehensive-release-audit-v1",
        "status": "PASS" if not errors else "REVIEW_REQUIRED",
        "lineage": {"raw_completed_runs": source_runs, "constructed_candidates": candidates, "source_exclusions": dict(excluded), "duplicate_run_ids_against_v22": sum(new_m["duplicate_run_ids_removed"].values()), "context_exclusions": sum(new_m["context_excluded"].values()), "v22_rows": old_m["total_rows"], "v23_increment": new_m["added_rows"], "v23_total": new_m["total_rows"]},
        "by_track_split_origin": {"/".join(k): v for k, v in sorted(by_track_split_origin.items())},
        "distributions": {origin: {d: pct(counters[origin][d]) for d in DIMS} for origin in counters},
        "lengths": {origin: {k: quantiles(v) for k, v in values.items()} for origin, values in lengths.items()},
        "schema_alignment": schema_alignment,
        "leakage": {"forbidden_pattern_hits": dict(leak_hits), "examples": leak_examples, "privileged_user_key_hits": dict(privileged_hits), "privileged_examples": privileged_examples, "builder_dynamic_secret_check": "PASS before candidate emission; builder aborts on any retained raw secret", "redaction_placeholder_counts": {k: dict(v) for k, v in redaction_placeholders.items()}},
        "dataset_version_metadata": dict(dataset_version),
        "validator_errors": validation_errors,
        "manifest_hashes": {"v22": sha256(a.v22 / "ASSEMBLY_MANIFEST.json"), "v23": sha256(a.v23 / "ASSEMBLY_MANIFEST.json"), "combined": sha256(a.combined / "COMBINED_MANIFEST.json")},
        "combined": combined_m,
        "errors": errors,
    }
    a.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# V23 Comprehensive Release Audit", "", f"**Status:** {report['status']}", "", "## Lineage", ""]
    for k, v in report["lineage"].items(): md.append(f"- {k}: {v}")
    md += ["", "## Contract and leakage", "", f"- Same system prompts: {schema_alignment['same_system_prompt_set']}", f"- Same user schema shapes: {schema_alignment['same_user_shape_set']}", f"- Same target schema shapes: {schema_alignment['same_target_shape_set']}", f"- SFT-visible forbidden-pattern hits: {sum(leak_hits.values())}", f"- Privileged user-key hits: {sum(privileged_hits.values())}", f"- Validator errors: {len(validation_errors)}", "", "See the JSON report for full distributions and length quantiles."]
    a.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "lineage": report["lineage"], "schema_alignment": schema_alignment, "leakage": report["leakage"], "errors": errors}, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
