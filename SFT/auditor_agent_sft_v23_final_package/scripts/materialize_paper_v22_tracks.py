from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


TRACK_SOURCES = {
    "marble_mab": r"D:\FIRST_COPILOT\plan_e\v20_marble_core_validation\context_filtered_dataset",
    "autogen_mab": r"D:\FIRST_COPILOT\.tmp_v22_all_inputs_20260817\v20_autogen_core_validation_v2\context_filtered_dataset",
    "marble_appworld": r"D:\FIRST_COPILOT\.tmp_v22_all_inputs_20260817\v20_appworld_marble_core_validation\context_filtered_dataset",
}
TRACKS = (*TRACK_SOURCES, "autogen_appworld")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\r\n"), json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--sealed-test",
        type=Path,
        default=Path(r"D:\FIRST_COPILOT\plan_e\v22_all_run\modernbert_sealed_test_source\test.jsonl"),
    )
    ap.add_argument(
        "--sealed-index",
        type=Path,
        default=Path(r"D:\FIRST_COPILOT\plan_e\v22_all_run\modernbert_sealed_test_source\track_index.jsonl"),
    )
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    test_rows = list(rows(args.sealed_test))
    test_index = [obj for _, obj in rows(args.sealed_index)]
    if len(test_rows) != len(test_index):
        raise RuntimeError("Sealed-test/index length mismatch")
    test_by_track = {track: [] for track in TRACKS}
    for (line, row), idx in zip(test_rows, test_index):
        if row["metadata"]["run_id"] != idx["run_id"]:
            raise RuntimeError(f"Sealed-test/index run mismatch: {idx['run_id']}")
        test_by_track[idx["track"]].append((line, row))

    manifest = {
        "version": "V22-paper-corpus-15931-frozen-v1",
        "definition": "Paper Section 5.2 corpus: 10,438 train + 2,954 validation + 2,539 sealed test",
        "tracks": {},
    }
    seen = set()
    total = 0
    for track in TRACKS:
        td = out / track
        td.mkdir()
        manifest["tracks"][track] = {}
        for split in ("train", "validation", "test"):
            if split == "test":
                records = test_by_track[track]
                source = str(args.sealed_test.resolve())
            elif track in TRACK_SOURCES:
                p = Path(TRACK_SOURCES[track]) / f"{split}.jsonl"
                records = list(rows(p))
                source = str(p.resolve())
            else:
                records = []
                source = None
            target = td / f"{split}.jsonl"
            counts = Counter()
            with target.open("w", encoding="utf-8", newline="\n") as f:
                for line, row in records:
                    rid = str(row["metadata"]["run_id"])
                    if rid in seen:
                        raise RuntimeError(f"Duplicate V22 run_id: {rid}")
                    seen.add(rid)
                    counts[row["metadata"]["verdict"]] += 1
                    f.write(line + "\n")
            n = len(records)
            total += n
            manifest["tracks"][track][split] = {
                "rows": n,
                "by_verdict": dict(counts),
                "sha256": digest(target),
                "source": source,
            }
    manifest["total_rows"] = total
    manifest["unique_run_ids"] = len(seen)
    if total != 15931:
        raise RuntimeError(f"Expected paper V22 total 15,931, got {total}")
    (out / "ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
