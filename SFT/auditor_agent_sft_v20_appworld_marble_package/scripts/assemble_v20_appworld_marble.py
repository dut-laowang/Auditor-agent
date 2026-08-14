from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_split(rows: list[dict], seed: int):
    rng = random.Random(seed)
    held_out = {}
    for scenario in sorted({str(r["metadata"]["scenario"]) for r in rows}):
        ids = sorted({int(r["metadata"]["sample_id"]) for r in rows if r["metadata"]["scenario"] == scenario})
        if len(ids) < 40:
            raise ValueError(f"Insufficient task groups: {scenario}={len(ids)}")
        rng.shuffle(ids)
        held_out[scenario] = {"test": sorted(ids[:16]), "validation": sorted(ids[16:34])}
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        meta = row["metadata"]
        sid = int(meta["sample_id"])
        assignment = held_out[str(meta["scenario"])]
        target = "test" if sid in assignment["test"] else "validation" if sid in assignment["validation"] else "train"
        splits[target].append(row)
    for split in splits.values():
        rng.shuffle(split)
    return splits, held_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_sft = Path(__file__).resolve().parents[2]
    v19 = repo_sft / "auditor_agent_sft_v19_qualityfix_package"
    builder = load(v19 / "scripts" / "build_v19_qualityfix_dataset.py", "v20_appworld_builder")
    v15_path = repo_sft / "auditor_agent_sft_v15_hq_current_package" / "scripts" / "build_v15_hq_current_dataset.py"
    adapter = load(Path(__file__).with_name("appworld_marble_observable_adapter.py"), "appworld_adapter")
    adapter.install(builder)

    # The transfer bundle places the manifest at its root while the canonical
    # MARBLE layout places it under merged/.  Normalize only this metadata file.
    canonical_manifest = args.source_root / "merged" / "run_manifest.jsonl"
    if not canonical_manifest.is_file():
        source_manifest = args.source_root / "run_manifest.jsonl"
        if not source_manifest.is_file():
            raise FileNotFoundError("run_manifest.jsonl missing from source bundle")
        shutil.copyfile(source_manifest, canonical_manifest)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_build = args.output_dir / "_source_build"
    builder.build(SimpleNamespace(
        source_root=args.source_root,
        output_dir=source_build,
        v15_builder=v15_path,
        source_archive_name="appworld_marble_random_3000_complete_20260814.tar.zst",
        source_type="appworld_marble_random_3000_v20",
        include_topology=[], exclude_topology=[], seed=args.seed, quiet=True,
    ))
    rows = builder.read_jsonl(source_build / "all.jsonl")
    for uid, row in enumerate(rows):
        sample_uid = f"v20_appworld_{uid:07d}"
        row["metadata"]["sample_uid"] = sample_uid
        user = json.loads(row["messages"][1]["content"])
        user["sample_uid"] = sample_uid
        row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    errors = builder.validate_samples(rows, builder.load_v15_builder(v15_path))
    if errors:
        raise RuntimeError(f"V20 AppWorld validation failed: {errors}")
    splits, held_out = group_split(rows, args.seed)
    groups = {name: {(r["metadata"]["scenario"], r["metadata"]["sample_id"]) for r in split} for name, split in splits.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise RuntimeError("Grouped split leakage detected")
    write_jsonl(args.output_dir / "all.jsonl", rows)
    for name, split in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    stats = {
        "schema": builder.SCHEMA,
        "version": "V20-appworld-marble-random-3000-observable-v1",
        "source_archive": "appworld_marble_random_3000_complete_20260814.tar.zst",
        "source_policy": "AppWorld×MARBLE; actual delivered_content for tool/message; MARBLE agent inputs and results; privileged instrumentation excluded",
        "split_policy": "seed-42 grouped by (scenario, sample_id): 18 validation and 16 sealed-test task IDs",
        "held_out_tasks": held_out,
        "files": {"all": builder.sample_stats(rows), **{name: builder.sample_stats(split) for name, split in splits.items()}},
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

