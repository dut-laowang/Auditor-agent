from __future__ import annotations

import argparse
import importlib.util
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_sft = Path(__file__).resolve().parents[2]
    v19 = repo_sft / "auditor_agent_sft_v19_qualityfix_package"
    builder_path = v19 / "scripts" / "build_v19_qualityfix_dataset.py"
    v15_path = repo_sft / "auditor_agent_sft_v15_hq_current_package" / "scripts" / "build_v15_hq_current_dataset.py"
    builder = load(builder_path, "v20_autogen_v19_builder")
    adapter = load(Path(__file__).with_name("autogen_observable_adapter.py"), "v20_autogen_observable_adapter")
    adapter.install(builder)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_build = args.output_dir / "_source_build"
    builder.build(
        SimpleNamespace(
            source_root=args.source_root,
            output_dir=source_build,
            v15_builder=v15_path,
            source_archive_name="autogen_native_complete_with_configs_20260813.tar.zst",
            source_type="autogen_native_v20_complete",
            include_topology=[],
            exclude_topology=[],
            seed=args.seed,
            quiet=True,
        )
    )
    rows = builder.read_jsonl(source_build / "all.jsonl")
    for uid, row in enumerate(rows):
        sample_uid = f"v20_autogen_{uid:07d}"
        row["metadata"]["sample_uid"] = sample_uid
        user = json.loads(row["messages"][1]["content"])
        user["sample_uid"] = sample_uid
        row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    errors = builder.validate_samples(rows, builder.load_v15_builder(v15_path))
    if errors:
        raise RuntimeError(f"V20 AutoGen validation failed: {errors}")

    # AutoGen has the same 10-task-per-scenario universe as V19 AutoGen, so the
    # original V19 grouped 2-validation/2-test split is retained exactly.
    train, validation, test, held_out = builder.group_split(rows, args.seed)
    for name, split in (("all", rows), ("train", train), ("validation", validation), ("test", test)):
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    stats = {
        "schema": builder.SCHEMA,
        "version": "V20-autogen-native-complete-20260813-observable-v2",
        "source_archive": "autogen_native_complete_with_configs_20260813.tar.zst",
        "source_policy": "AutoGen single+dual; actual delivered_content from original trajectory logs only; privileged attack instrumentation excluded; strict source-grounded observable-evidence gate; V20 output schema unchanged",
        "split_policy": "V19 seed-42 grouping by (scenario, sample_id): 2 validation and 2 sealed-test task IDs per scenario",
        "held_out_tasks": held_out,
        "files": {name: builder.sample_stats(split) for name, split in (("all", rows), ("train", train), ("validation", validation), ("test", test))},
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
