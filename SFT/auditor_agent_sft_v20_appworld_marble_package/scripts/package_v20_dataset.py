from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    args = parser.parse_args()
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    archive = args.bundle_dir / "dataset_jsonl.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in FILES:
            zf.write(args.data_dir / name, arcname=name)
    for name in ("stats.json", "quality_audit.json", "appworld_observable_quality.json"):
        shutil.copy2(args.data_dir / name, args.bundle_dir / name)
    stats = json.loads((args.data_dir / "stats.json").read_text(encoding="utf-8"))
    track_manifest = {
        "schema": stats["schema"],
        "allowed_source_types": ["appworld_marble_random_3000_v20"],
        "split_assignment": stats["split_policy"],
        "stats": {
            name: stats["files"][name]
            for name in ("train", "validation", "test")
        },
        "sha256": {name: sha256(args.data_dir / name) for name in FILES},
    }
    (args.bundle_dir / "track_manifest.json").write_text(
        json.dumps(track_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [f"{sha256(args.data_dir / name)}  {name}" for name in FILES]
    lines.extend(
        f"{sha256(args.bundle_dir / name)}  {name}"
        for name in ("stats.json", "quality_audit.json", "appworld_observable_quality.json", "track_manifest.json")
    )
    (args.bundle_dir / "SHA256SUMS").write_text("\n".join(sorted(lines)) + "\n", encoding="ascii")
    (args.bundle_dir / "TRANSPORT_SHA256").write_text(
        f"{sha256(archive)}  dataset_jsonl.zip\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
