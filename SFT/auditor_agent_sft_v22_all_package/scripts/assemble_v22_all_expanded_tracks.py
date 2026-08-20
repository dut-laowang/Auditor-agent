from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")
NEW_TRACK_ARGS = {
    "autogen_mab": "new_autogen_mab",
    "marble_appworld": "new_marble_appworld",
    "autogen_appworld": "new_autogen_appworld",
}


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc


def zipped_jsonl(path: Path, member: str):
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        for number, raw in enumerate(handle, 1):
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}!{member}:{number}: {exc}") from exc


def task_key(row: dict) -> tuple[str, str, str]:
    meta = row["metadata"]
    family = "appworld" if str(meta.get("scenario")) == "appworld" else "multiagentbench"
    return family, str(meta.get("scenario")), str(meta.get("sample_id"))


def set_uid(row: dict) -> None:
    run_id = str(row["metadata"]["run_id"])
    uid = "v22all_" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    row["metadata"]["sample_uid"] = uid
    user = json.loads(row["messages"][1]["content"])
    user["sample_uid"] = uid
    row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble leakage-safe expanded V22 2x2 tracks.")
    parser.add_argument("--existing-source-root", required=True, type=Path)
    parser.add_argument("--marble-mab-test-zip", required=True, type=Path)
    parser.add_argument("--autogen-mab-test-zip", required=True, type=Path)
    parser.add_argument("--marble-appworld-test-zip", required=True, type=Path)
    parser.add_argument("--new-autogen-mab", required=True, type=Path)
    parser.add_argument("--new-marble-appworld", required=True, type=Path)
    parser.add_argument("--new-autogen-appworld", required=True, type=Path)
    parser.add_argument("--qwen-gate-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, list[dict]]] = {
        track: {split: [] for split in ("train", "validation", "test")} for track in TRACKS
    }
    split_by_task: dict[tuple[str, str, str], str] = {}
    seen_ids: dict[str, tuple[str, str]] = {}
    duplicate_new = Counter()
    context_excluded: set[str] = set()
    if args.qwen_gate_manifest:
        gate = json.loads(args.qwen_gate_manifest.read_text(encoding="utf-8"))
        for split in gate.get("splits", {}).values():
            context_excluded.update(str(item["run_id"]) for item in split.get("dropped", []))

    source = args.existing_source_root.resolve()
    for split in ("train", "validation"):
        data = list(jsonl(source / "base_dataset" / f"{split}.jsonl"))
        index = list(jsonl(source / "track_index" / f"{split}.jsonl"))
        if len(data) != len(index):
            raise RuntimeError(f"Existing {split} data/index length mismatch")
        for row, idx in zip(data, index):
            if row["metadata"]["run_id"] != idx["run_id"]:
                raise RuntimeError(f"Existing {split} data/index order mismatch")
            track = idx["track"]
            rows[track][split].append(row)
            split_by_task.setdefault(task_key(row), split)

    test_inputs = {
        "marble_mab": args.marble_mab_test_zip,
        "autogen_mab": args.autogen_mab_test_zip,
        "marble_appworld": args.marble_appworld_test_zip,
    }
    for track, path in test_inputs.items():
        for row in zipped_jsonl(path.resolve(), "test.jsonl"):
            rows[track]["test"].append(row)
            split_by_task.setdefault(task_key(row), "test")

    for track in TRACKS:
        for split in ("train", "validation", "test"):
            for row in rows[track][split]:
                run_id = str(row["metadata"]["run_id"])
                if run_id in seen_ids:
                    raise RuntimeError(f"Duplicate original run_id: {run_id}")
                seen_ids[run_id] = (track, split)
                assigned = split_by_task.get(task_key(row))
                if assigned != split:
                    raise RuntimeError(f"Original task split conflict: {run_id}: {assigned} != {split}")

    for track, arg_name in NEW_TRACK_ARGS.items():
        for row in jsonl(getattr(args, arg_name).resolve()):
            run_id = str(row["metadata"]["run_id"])
            if run_id in context_excluded:
                continue
            if run_id in seen_ids:
                duplicate_new[track] += 1
                continue
            split = split_by_task.get(task_key(row))
            if split is None:
                raise RuntimeError(f"New row has unknown task group: {track}/{run_id}")
            rows[track][split].append(row)
            seen_ids[run_id] = (track, split)

    split_tasks: dict[str, set] = defaultdict(set)
    split_inputs: dict[str, set] = defaultdict(set)
    all_uids = set()
    summary = {
        "tracks": {},
        "duplicates_removed": dict(duplicate_new),
        "qwen_context_excluded": len(context_excluded),
    }
    for track in TRACKS:
        track_dir = output / track
        track_dir.mkdir()
        summary["tracks"][track] = {}
        for split in ("train", "validation", "test"):
            path = track_dir / f"{split}.jsonl"
            counts = Counter()
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows[track][split]:
                    set_uid(row)
                    uid = row["metadata"]["sample_uid"]
                    if uid in all_uids:
                        raise RuntimeError(f"Duplicate deterministic sample_uid: {uid}")
                    all_uids.add(uid)
                    split_tasks[split].add(task_key(row))
                    split_inputs[split].add(hashlib.sha256(row["messages"][1]["content"].encode()).hexdigest())
                    counts[row["metadata"]["verdict"]] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["tracks"][track][split] = {
                "rows": sum(counts.values()), "by_verdict": dict(counts), "sha256": sha256(path)
            }

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}_task_group"] = len(split_tasks[left] & split_tasks[right])
        overlaps[f"{left}_{right}_exact_user_input"] = len(split_inputs[left] & split_inputs[right])
    if any(overlaps.values()):
        raise RuntimeError(f"Expanded split leakage: {overlaps}")
    summary.update({
        "version": "V22-ALL-expanded-2x2-tracks-v1",
        "total_rows": len(seen_ids),
        "unique_run_ids": len(seen_ids),
        "unique_sample_uids": len(all_uids),
        "split_overlap": overlaps,
    })
    (output / "ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
