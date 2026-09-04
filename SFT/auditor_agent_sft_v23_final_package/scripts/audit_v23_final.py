from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(row: dict) -> str:
    user = json.loads(row["messages"][1]["content"])
    user.pop("sample_uid", None)
    value = json.dumps(user, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def task_key(row: dict) -> tuple[str, str, str]:
    m = row["metadata"]
    family = "appworld" if m.get("scenario") == "appworld" else "multiagentbench"
    return family, str(m.get("scenario")), str(m.get("sample_id"))


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("v22_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module.validate_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v22", required=True, type=Path)
    ap.add_argument("--v23", required=True, type=Path)
    ap.add_argument("--combined", required=True, type=Path)
    ap.add_argument("--validator", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    old_manifest = json.loads((a.v22 / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((a.v23 / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    combined_manifest = json.loads((a.combined / "COMBINED_MANIFEST.json").read_text(encoding="utf-8"))
    validate_row = load_validator(a.validator)
    errors = []
    ids, uids = {}, {}
    hashes = defaultdict(list)
    split_tasks, split_inputs = defaultdict(set), defaultdict(set)
    totals, verdicts = Counter(), Counter()
    v22_prefix_checks = {}
    for track in TRACKS:
        for split in SPLITS:
            old_path, new_path = a.v22 / track / f"{split}.jsonl", a.v23 / track / f"{split}.jsonl"
            expected_old = old_manifest["tracks"][track][split]
            expected_new = new_manifest["tracks"][track][split]
            old_bytes = old_path.read_bytes()
            new_bytes = new_path.read_bytes()
            prefix_ok = new_bytes.startswith(old_bytes)
            v22_prefix_checks[f"{track}/{split}"] = prefix_ok
            if not prefix_ok:
                errors.append(f"V22 byte prefix changed: {track}/{split}")
            if sha256(old_path) != expected_old["sha256"]:
                errors.append(f"V22 source hash mismatch: {track}/{split}")
            if sha256(new_path) != expected_new["sha256"]:
                errors.append(f"V23 track hash mismatch: {track}/{split}")
            rows = 0
            with new_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    rows += 1
                    row = json.loads(line)
                    rid = str(row.get("metadata", {}).get("run_id", ""))
                    uid = str(row.get("metadata", {}).get("sample_uid", ""))
                    if rid in ids:
                        errors.append(f"duplicate run_id: {rid}")
                    ids[rid] = (track, split)
                    if uid in uids:
                        errors.append(f"duplicate sample_uid: {uid}")
                    uids[uid] = rid
                    hashes[canonical_hash(row)].append({"run_id": rid, "track": track, "split": split, "position": line_no, "origin": "V22" if line_no <= expected_old["rows"] else "V23"})
                    split_tasks[split].add(task_key(row))
                    split_inputs[split].add(canonical_hash(row))
                    verdicts[row["metadata"]["verdict"]] += 1
                    problems = validate_row(row, track, split)
                    if problems:
                        errors.extend(problems)
            totals[split] += rows
            if rows != expected_new["rows"]:
                errors.append(f"row mismatch: {track}/{split}: {rows}")
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}_task"] = len(split_tasks[left] & split_tasks[right])
        overlaps[f"{left}_{right}_input"] = len(split_inputs[left] & split_inputs[right])
    if any(overlaps.values()):
        errors.append(f"split leakage: {overlaps}")
    canonical_duplicates = [group for group in hashes.values() if len(group) > 1]
    new_duplicate_groups = [g for g in canonical_duplicates if any(x["origin"] == "V23" for x in g)]
    if new_duplicate_groups:
        errors.append(f"new canonical duplicate groups: {len(new_duplicate_groups)}")
    for split in SPLITS:
        data = a.combined / f"{split}.jsonl"
        index = a.combined / f"{split}_track_index.jsonl"
        cm = combined_manifest["splits"][split]
        if sha256(data) != cm["sha256"] or sha256(index) != cm["index_sha256"]:
            errors.append(f"combined hash mismatch: {split}")
        if totals[split] != cm["rows"]:
            errors.append(f"combined count mismatch: {split}")
    report = {
        "version": "V23-final-quality-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "parent_rows": old_manifest["total_rows"],
        "total_rows": sum(totals.values()),
        "added_rows": sum(totals.values()) - old_manifest["total_rows"],
        "by_split": dict(totals),
        "by_verdict": dict(verdicts),
        "unique_run_ids": len(ids),
        "unique_sample_uids": len(uids),
        "unique_canonical_inputs": len(hashes),
        "historical_canonical_duplicate_groups": len(canonical_duplicates),
        "new_canonical_duplicate_groups": len(new_duplicate_groups),
        "canonical_duplicate_details": canonical_duplicates,
        "v22_byte_prefix_unchanged": all(v22_prefix_checks.values()),
        "v22_prefix_checks": v22_prefix_checks,
        "split_overlap": overlaps,
        "schema_and_semantic_rows_checked": sum(totals.values()),
        "errors": errors[:1000],
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
