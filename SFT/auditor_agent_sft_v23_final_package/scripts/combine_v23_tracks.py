from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TRACKS = ("marble_mab", "autogen_mab", "marble_appworld", "autogen_appworld")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize combined V23 data and track indices.")
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    tracks = args.tracks.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assembly = json.loads((tracks / "ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    manifest = {"version": "V23-ALL-expanded-2x2-combined-final-v1", "splits": {}}
    for split in ("train", "validation", "test"):
        data_path = output / f"{split}.jsonl"
        index_path = output / f"{split}_track_index.jsonl"
        count = 0
        with data_path.open("w", encoding="utf-8", newline="\n") as data_out, index_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as index_out:
            for track in TRACKS:
                source = tracks / track / f"{split}.jsonl"
                expected = assembly["tracks"][track][split]
                if sha256(source) != expected["sha256"]:
                    raise RuntimeError(f"Track hash mismatch: {source}")
                track_count = 0
                with source.open(encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        data_out.write(line.rstrip("\r\n") + "\n")
                        index_out.write(
                            json.dumps(
                                {
                                    "run_id": row["metadata"]["run_id"],
                                    "track": track,
                                    "split": split,
                                    "verdict": row["metadata"]["verdict"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
                        track_count += 1
                if track_count != expected["rows"]:
                    raise RuntimeError(f"Track row-count mismatch: {track}/{split}")
        manifest["splits"][split] = {
            "rows": count,
            "sha256": sha256(data_path),
            "index_sha256": sha256(index_path),
        }
    (output / "COMBINED_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
