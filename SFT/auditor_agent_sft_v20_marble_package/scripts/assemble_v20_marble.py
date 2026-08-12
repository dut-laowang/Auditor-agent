from __future__ import annotations

import argparse
import importlib.util
import json
import random
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


def group_split_v20(rows: list[dict], seed: int):
    """V19 grouped split scaled to the larger 90-task-per-scenario source.

    Keep the same no-leakage unit, (scenario, sample_id), while assigning 18
    validation and 16 sealed-test task IDs per scenario. This preserves roughly
    the V19 validation/test proportions instead of shrinking each split to ~2%.
    """
    rng = random.Random(seed)
    held_out = {}
    for scenario in sorted({str(r["metadata"]["scenario"]) for r in rows}):
        ids = sorted(
            {
                int(r["metadata"]["sample_id"])
                for r in rows
                if r["metadata"]["scenario"] == scenario
            }
        )
        if len(ids) < 40:
            raise ValueError(f"Insufficient task groups for V20 split: {scenario}={len(ids)}")
        rng.shuffle(ids)
        held_out[scenario] = {
            "test": sorted(ids[:16]),
            "validation": sorted(ids[16:34]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_sft = Path(__file__).resolve().parents[2]
    v19 = repo_sft / "auditor_agent_sft_v19_qualityfix_package"
    builder_path = v19 / "scripts" / "build_v19_qualityfix_dataset.py"
    v15_path = (
        repo_sft
        / "auditor_agent_sft_v15_hq_current_package"
        / "scripts"
        / "build_v15_hq_current_dataset.py"
    )
    builder = load(builder_path, "v20_v19_builder")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_build = args.output_dir / "_source_build"
    builder.build(
        SimpleNamespace(
            source_root=args.source_root,
            output_dir=source_build,
            v15_builder=v15_path,
            source_archive_name="marble_random_10665_trajectories_configs_labels.tar.zst",
            source_type="marble_random_10665_v20",
            include_topology=[],
            exclude_topology=[],
            seed=args.seed,
            quiet=True,
        )
    )
    rows = builder.read_jsonl(source_build / "all.jsonl")
    for uid, row in enumerate(rows):
        row["metadata"]["sample_uid"] = f"v20_{uid:07d}"
        user = json.loads(row["messages"][1]["content"])
        user["sample_uid"] = f"v20_{uid:07d}"
        row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    errors = builder.validate_samples(rows, builder.load_v15_builder(v15_path))
    if errors:
        raise RuntimeError(f"V20 validation failed: {errors}")
    train, validation, test, held_out = group_split_v20(rows, args.seed)
    for name, split in (("all", rows), ("train", train), ("validation", validation), ("test", test)):
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    stats = {
        "schema": builder.SCHEMA,
        "version": "V20-marble-random-10665",
        "source_archive": "marble_random_10665_trajectories_configs_labels.tar.zst",
        "source_policy": "MARBLE only; all four native topologies; V19 observable-input and label policy unchanged",
        "split_policy": "V19 seed-42 grouping by (scenario, sample_id), scaled to 18 validation and 16 sealed-test task IDs per scenario for the larger V20 task universe",
        "held_out_tasks": held_out,
        "files": {name: builder.sample_stats(split) for name, split in (("all", rows), ("train", train), ("validation", validation), ("test", test))},
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
