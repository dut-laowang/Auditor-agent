from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("track_dir", type=Path)
    args = parser.parse_args()
    archive = args.track_dir / "dataset_jsonl.zip"
    transport_line = (args.track_dir / "TRANSPORT_SHA256").read_text(encoding="ascii").strip()
    transport_hash, transport_name = transport_line.split(None, 1)
    if transport_name.strip() != archive.name or sha256(archive) != transport_hash:
        raise RuntimeError(f"Transport archive SHA-256 mismatch: {archive}")
    manifest = json.loads((args.track_dir / "track_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["sha256"]
    with zipfile.ZipFile(archive) as handle:
        if sorted(handle.namelist()) != sorted(expected):
            raise RuntimeError(f"Unexpected ZIP members: {handle.namelist()}")
        for name in expected:
            path = args.track_dir / name
            # Restored JSONL files are generated transport artifacts and may be
            # left behind by an older Git revision. Replace a stale copy from the
            # newly verified ZIP instead of failing before it can be refreshed.
            if not path.is_file() or sha256(path) != expected[name]:
                handle.extract(name, args.track_dir)
            actual = sha256(path)
            if actual != expected[name]:
                raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected[name]}")
    print(json.dumps({"track_dir": str(args.track_dir), "verified": sorted(expected)}))


if __name__ == "__main__":
    main()
