from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


TRACKS = {
    "marble_only": {"marble_legacy_nonstar", "marble_star_fixed"},
    "autogen_only": {"autogen_native"},
    "mixed": {"marble_legacy_nonstar", "marble_star_fixed", "autogen_native"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_track(source: Path, destination: Path, allowed: set[str]) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    stats = {}
    for split in ("train", "validation", "test"):
        counts = Counter()
        output = destination / f"{split}.jsonl"
        with (source / f"{split}.jsonl").open(encoding="utf-8-sig") as reader, output.open("w", encoding="utf-8") as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row["metadata"]
                if meta["source_type"] not in allowed:
                    continue
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["total"] += 1
                counts[f"verdict::{meta['verdict']}"] += 1
                counts[f"source::{meta['source_type']}"] += 1
                counts[f"topology::{meta['topology']}"] += 1
        stats[split] = dict(counts)
    manifest = {
        "schema": "Graph-grounded-Candidate-SFT/v13",
        "allowed_source_types": sorted(allowed),
        "split_assignment": "inherited unchanged from the unified V19 grouped split",
        "stats": stats,
        "sha256": {
            f"{split}.jsonl": sha256(destination / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
    }
    (destination / "track_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-data-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, allowed in TRACKS.items():
        destination = args.output_root / name
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite track: {destination}")
        summary[name] = write_track(args.mixed_data_dir, destination, allowed)
    (args.output_root / "three_track_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
