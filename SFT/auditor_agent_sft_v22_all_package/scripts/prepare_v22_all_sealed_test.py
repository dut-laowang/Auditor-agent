from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from prepare_v22_all_source import task_key, validate_row


TRACKS = {
    "marble_mab": 1522,
    "autogen_mab": 624,
    "marble_appworld": 393,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_test_member(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "test.jsonl"]
        if members != ["test.jsonl"]:
            raise RuntimeError(f"{path}: expected exactly the root test.jsonl member, got {members}")
        text = archive.read(members[0]).decode("utf-8")
    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip()]
    return lines, [json.loads(line) for line in lines]


def infer_track(row: dict) -> str:
    source_type = str(row.get("metadata", {}).get("source_type", ""))
    if "appworld" in source_type:
        return "marble_appworld"
    if "autogen" in source_type:
        return "autogen_mab"
    return "marble_mab"


def verify_existing(output: Path) -> bool:
    manifest_path = output / "SEALED_TEST_MANIFEST.json"
    data_path = output / "test.jsonl"
    index_path = output / "track_index.jsonl"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("rows") != sum(TRACKS.values())
        or manifest.get("test_sha256") != sha256(data_path)
        or manifest.get("track_index_sha256") != sha256(index_path)
    ):
        raise RuntimeError("Existing sealed-test preparation does not match its manifest")
    print(json.dumps({"status": "PASS", "reused": True, **manifest}, ensure_ascii=False, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the final, labeled V22-ALL ModernBERT test exactly once.")
    parser.add_argument("--marble-mab-zip", required=True, type=Path)
    parser.add_argument("--autogen-mab-zip", required=True, type=Path)
    parser.add_argument("--marble-appworld-zip", required=True, type=Path)
    parser.add_argument("--reference-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and verify_existing(output):
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing incomplete/non-empty sealed-test directory: {output}")
    if output.exists():
        output.rmdir()

    inputs = {
        "marble_mab": args.marble_mab_zip.resolve(),
        "autogen_mab": args.autogen_mab_zip.resolve(),
        "marble_appworld": args.marble_appworld_zip.resolve(),
    }
    reference_rows = []
    for split in ("train", "validation"):
        path = args.reference_data / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        reference_rows.extend(read_jsonl(path))
    reference_ids = {row["metadata"]["run_id"] for row in reference_rows}
    reference_inputs = {sha256_text(row["messages"][1]["content"]) for row in reference_rows}
    reference_tasks = {task_key(row, infer_track(row)) for row in reference_rows}

    staged = output.with_name(output.name + ".staging")
    if staged.exists():
        raise RuntimeError(f"Refusing stale sealed-test staging directory: {staged}")
    staged.mkdir(parents=True)
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    track_counts: dict[str, int] = {}
    try:
        with (staged / "test.jsonl").open("w", encoding="utf-8", newline="\n") as data_out, \
             (staged / "track_index.jsonl").open("w", encoding="utf-8", newline="\n") as index_out:
            for track, archive in inputs.items():
                lines, rows = read_test_member(archive)
                track_counts[track] = len(rows)
                if len(rows) != TRACKS[track]:
                    problems.append(f"{track}: expected {TRACKS[track]} test rows, got {len(rows)}")
                for line, row in zip(lines, rows):
                    problems.extend(validate_row(row, track, "test"))
                    run_id = str(row.get("metadata", {}).get("run_id") or "")
                    user_hash = sha256_text(row["messages"][1]["content"])
                    if not run_id:
                        problems.append(f"{track}: missing run_id")
                    if run_id in seen_ids or run_id in reference_ids:
                        problems.append(f"{track}/{run_id}: duplicate or train/validation run_id overlap")
                    if user_hash in seen_inputs or user_hash in reference_inputs:
                        problems.append(f"{track}/{run_id}: exact user input overlaps another split/track")
                    if task_key(row, track) in reference_tasks:
                        problems.append(f"{track}/{run_id}: task group overlaps train/validation")
                    seen_ids.add(run_id)
                    seen_inputs.add(user_hash)
                    data_out.write(line + "\n")
                    index_out.write(json.dumps({
                        "run_id": run_id,
                        "track": track,
                        "split": "test",
                        "verdict": row["metadata"]["verdict"],
                    }, ensure_ascii=False) + "\n")
        if problems:
            raise RuntimeError(f"Sealed-test quality gate failed ({len(problems)}): {problems[:10]}")
        data_path = staged / "test.jsonl"
        index_path = staged / "track_index.jsonl"
        manifest = {
            "version": "V22-ALL-sealed-test-v1",
            "status": "PASS",
            "rows": sum(track_counts.values()),
            "tracks": track_counts,
            "test_sha256": sha256(data_path),
            "track_index_sha256": sha256(index_path),
            "all_rows_semantically_validated": True,
            "train_validation_run_id_overlap": 0,
            "train_validation_exact_user_input_overlap": 0,
            "train_validation_task_group_overlap": 0,
            "labels_present_for_scoring_only": True,
            "model_input_mode": "user",
            "sealed_test_accessed": True,
        }
        (staged / "SEALED_TEST_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staged.rename(output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        failure = staged / "SEALED_TEST_PREPARATION_FAILED.json"
        failure.write_text(json.dumps({"status": "FAIL", "problems": problems}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
