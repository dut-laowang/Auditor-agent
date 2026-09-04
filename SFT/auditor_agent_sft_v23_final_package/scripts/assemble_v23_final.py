from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")
POOL_TRACKS = {
    "mab_marble_7000": "marble_mab",
    "mab_marble_attack_only_1000_20260828": "marble_mab",
    "mab_marble_attack_only_2000_20260828": "marble_mab",
    "mab_autogen_4000": "autogen_mab",
    "appworld_marble_4000": "marble_appworld",
    "appworld_marble_attack_only_1000_20260828": "marble_appworld",
    "appworld_marble_attack_only_2000_20260828": "marble_appworld",
    "appworld_autogen_4000": "autogen_appworld",
}
POOL_ORDER = tuple(POOL_TRACKS)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line.rstrip("\r\n"), json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_key(row: dict) -> tuple[str, str, str]:
    meta = row["metadata"]
    scenario = str(meta.get("scenario"))
    family = "appworld" if scenario == "appworld" else "multiagentbench"
    return family, scenario, str(meta.get("sample_id"))


def canonical_user_hash(row: dict) -> str:
    user = json.loads(row["messages"][1]["content"])
    user.pop("sample_uid", None)
    canonical = json.dumps(user, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_v23_uid(row: dict) -> None:
    run_id = str(row["metadata"]["run_id"])
    uid = "v23_" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    row["metadata"]["sample_uid"] = uid
    # Keep the expanded release schema identical to frozen V22.  Lineage is
    # recorded in ASSEMBLY_MANIFEST.json and by the V22 byte-prefix boundary,
    # never by adding a new per-row field that old loaders have not seen.
    row["metadata"].pop("dataset_version", None)
    user = json.loads(row["messages"][1]["content"])
    user["sample_uid"] = uid
    row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append leakage-safe V23 candidates to frozen V22 tracks.")
    parser.add_argument("--v22-tracks", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exclude-run-ids", type=Path)
    args = parser.parse_args()

    v22 = args.v22_tracks.resolve()
    candidates = args.candidates.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    excluded_context = set()
    if args.exclude_run_ids:
        payload = json.loads(args.exclude_run_ids.read_text(encoding="utf-8"))
        excluded_context = {str(value) for value in payload.get("run_ids", [])}

    v22_manifest = json.loads((v22 / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    seen_ids: dict[str, tuple[str, str]] = {}
    seen_inputs: dict[str, str] = {}
    split_by_task: dict[tuple[str, str, str], str] = {}
    v22_lines: dict[str, dict[str, list[str]]] = defaultdict(dict)
    additions: dict[str, dict[str, list[dict]]] = {
        track: {split: [] for split in ("train", "validation", "test")} for track in TRACKS
    }

    for track in TRACKS:
        for split in ("train", "validation", "test"):
            path = v22 / track / f"{split}.jsonl"
            expected = v22_manifest["tracks"][track][split]
            if sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Frozen V22 hash mismatch: {path}")
            records = list(read_jsonl(path))
            if len(records) != expected["rows"]:
                raise RuntimeError(f"Frozen V22 count mismatch: {path}")
            v22_lines[track][split] = [line for line, _ in records]
            for _, row in records:
                run_id = str(row["metadata"]["run_id"])
                if run_id in seen_ids:
                    raise RuntimeError(f"Duplicate run_id inside V22: {run_id}")
                seen_ids[run_id] = (track, split)
                key = task_key(row)
                previous = split_by_task.setdefault(key, split)
                if previous != split:
                    raise RuntimeError(f"Frozen V22 task split conflict: {key}: {previous}/{split}")
                seen_inputs.setdefault(canonical_user_hash(row), run_id)

    duplicate_run_ids = Counter()
    duplicate_inputs = Counter()
    context_excluded = Counter()
    accepted_by_pool = Counter()
    rejected_examples = []
    for pool in POOL_ORDER:
        track = POOL_TRACKS[pool]
        path = candidates / pool / "all.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing candidate pool: {path}")
        for _, row in read_jsonl(path):
            run_id = str(row["metadata"]["run_id"])
            if run_id in excluded_context:
                context_excluded[pool] += 1
                continue
            if run_id in seen_ids:
                duplicate_run_ids[pool] += 1
                if len(rejected_examples) < 100:
                    rejected_examples.append({"pool": pool, "run_id": run_id, "reason": "duplicate_run_id"})
                continue
            input_hash = canonical_user_hash(row)
            if input_hash in seen_inputs:
                duplicate_inputs[pool] += 1
                if len(rejected_examples) < 100:
                    rejected_examples.append({
                        "pool": pool,
                        "run_id": run_id,
                        "reason": "duplicate_canonical_user_input",
                        "matches_run_id": seen_inputs[input_hash],
                    })
                continue
            split = split_by_task.get(task_key(row))
            if split is None:
                raise RuntimeError(f"Unknown V22 task group for {pool}/{run_id}: {task_key(row)}")
            assign_v23_uid(row)
            additions[track][split].append(row)
            seen_ids[run_id] = (track, split)
            seen_inputs[input_hash] = run_id
            accepted_by_pool[pool] += 1

    manifest = {
        "version": "V23-ALL-expanded-2x2-tracks-v1",
        "parent_version": v22_manifest["version"],
        "parent_rows": v22_manifest["total_rows"],
        "tracks": {},
        "candidate_pools": list(POOL_ORDER),
        "accepted_by_pool": dict(accepted_by_pool),
        "duplicate_run_ids_removed": dict(duplicate_run_ids),
        "duplicate_canonical_inputs_removed": dict(duplicate_inputs),
        "context_excluded": dict(context_excluded),
        "rejected_examples": rejected_examples,
    }
    split_tasks: dict[str, set] = defaultdict(set)
    split_inputs: dict[str, set] = defaultdict(set)
    total_rows = 0
    for track in TRACKS:
        track_dir = output / track
        track_dir.mkdir()
        manifest["tracks"][track] = {}
        for split in ("train", "validation", "test"):
            path = track_dir / f"{split}.jsonl"
            counts = Counter()
            old_count = len(v22_lines[track][split])
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for line in v22_lines[track][split]:
                    row = json.loads(line)
                    handle.write(line + "\n")
                    counts[row["metadata"]["verdict"]] += 1
                    split_tasks[split].add(task_key(row))
                    split_inputs[split].add(canonical_user_hash(row))
                for row in additions[track][split]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[row["metadata"]["verdict"]] += 1
                    split_tasks[split].add(task_key(row))
                    split_inputs[split].add(canonical_user_hash(row))
            rows = sum(counts.values())
            total_rows += rows
            manifest["tracks"][track][split] = {
                "rows": rows,
                "v22_rows_unchanged": old_count,
                "v23_added_rows": rows - old_count,
                "by_verdict": dict(counts),
                "sha256": sha256(path),
            }

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}_task_group"] = len(split_tasks[left] & split_tasks[right])
        overlaps[f"{left}_{right}_canonical_user_input"] = len(split_inputs[left] & split_inputs[right])
    if any(overlaps.values()):
        raise RuntimeError(f"V23 split leakage: {overlaps}")
    manifest.update(
        {
            "total_rows": total_rows,
            "added_rows": total_rows - v22_manifest["total_rows"],
            "unique_run_ids": len(seen_ids),
            "unique_canonical_user_inputs": len(seen_inputs),
            "split_overlap": overlaps,
        }
    )
    (output / "ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
