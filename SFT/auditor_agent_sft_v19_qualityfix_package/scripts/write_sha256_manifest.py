from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", default="SHA256SUMS")
    args = parser.parse_args()
    output = args.root / args.output
    files = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path != output and "_source_builds" not in path.parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(args.root).as_posix()}" for path in files]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
