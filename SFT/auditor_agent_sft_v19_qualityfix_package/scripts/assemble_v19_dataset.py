from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("v19_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-marble-root", required=True, type=Path)
    parser.add_argument("--star-fixed-root", required=True, type=Path)
    parser.add_argument("--autogen-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--v15-builder", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists():
        unexpected = [path.name for path in args.output_dir.iterdir() if path.name != "_source_builds"]
        if unexpected:
            raise RuntimeError(
                f"Refusing to overwrite finalized output files in {args.output_dir}: {unexpected}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder_path = Path(__file__).with_name("build_v19_qualityfix_dataset.py")
    builder = load_builder(builder_path)
    sources = [
        (
            "marble_legacy_nonstar",
            args.old_marble_root,
            [],
            ["star"],
            "weekend_fresh_single1_10_dual1_5_bundle.tar.zst",
        ),
        (
            "marble_star_fixed",
            args.star_fixed_root,
            ["star"],
            [],
            "weekend_fresh_star_fixed.zip",
        ),
        (
            "autogen_native",
            args.autogen_root,
            [],
            [],
            "autogen_native_minimal_20260806.tar.zst",
        ),
    ]

    combined: list[dict] = []
    source_stats: dict[str, dict] = {}
    work_root = args.output_dir / "_source_builds"
    for source_type, source_root, include, exclude, archive_name in sources:
        source_output = work_root / source_type
        if not (source_output / "all.jsonl").is_file() or not (source_output / "stats.json").is_file():
            builder.build(
                SimpleNamespace(
                    source_root=source_root,
                    output_dir=source_output,
                    v15_builder=args.v15_builder,
                    source_archive_name=archive_name,
                    source_type=source_type,
                    include_topology=include,
                    exclude_topology=exclude,
                    seed=args.seed,
                    quiet=True,
                )
            )
        rows = read_jsonl(source_output / "all.jsonl")
        combined.extend(rows)
        source_stats[source_type] = json.loads(
            (source_output / "stats.json").read_text(encoding="utf-8")
        )["files"]["all"]

    seen: set[tuple[str, str]] = set()
    for row in combined:
        meta = row["metadata"]
        key = (str(meta["source_type"]), str(meta["run_id"]))
        if key in seen:
            raise RuntimeError(f"Duplicate source/run pair: {key}")
        seen.add(key)
    for uid, row in enumerate(combined):
        builder.stable_uid(row, uid)

    validation_errors = builder.validate_samples(combined, builder.load_v15_builder(args.v15_builder))
    if validation_errors:
        raise RuntimeError(f"Combined validation failed: {validation_errors}")

    train, validation, test, held_out = builder.group_split(combined, args.seed)
    write_jsonl(args.output_dir / "all.jsonl", combined)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "test.jsonl", test)
    builder.make_manual_review_queue(
        combined,
        args.output_dir / "manual_review_queue_200_v19.json",
        args.seed,
        count=200,
    )
    stats = {
        "schema": builder.SCHEMA,
        "version": "V19-qualityfix",
        "source_policy": {
            "marble_legacy": "chain/graph/tree retained; legacy star excluded",
            "marble_star": "star_fixed replaces legacy star",
            "autogen": "native AutoGen added as a cross-framework source",
        },
        "split_policy": (
            "Grouped by (scenario, sample_id) across all frameworks and topologies; "
            "test is not an input to training or model selection."
        ),
        "held_out_tasks": held_out,
        "sources": source_stats,
        "files": {
            "all": builder.sample_stats(combined),
            "train": builder.sample_stats(train),
            "validation": builder.sample_stats(validation),
            "test": builder.sample_stats(test),
        },
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
