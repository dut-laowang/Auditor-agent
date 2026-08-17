from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def run(*parts: object) -> None:
    command = [str(part) for part in parts]
    print(json.dumps({"run": command}, ensure_ascii=False))
    subprocess.run(command, check=True)


def find_source_root(root: Path) -> Path:
    candidates = []
    for manifest in root.rglob("run_manifest.jsonl"):
        candidate = manifest.parent.parent if manifest.parent.name == "merged" else manifest.parent
        if (candidate / "merged").is_dir() or (candidate / "trajectories").is_dir():
            candidates.append(candidate)
    unique = sorted(set(candidates), key=lambda value: (len(value.parts), str(value)))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one AppWorld x MARBLE source root, found: {unique}")
    return unique[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a leakage-safe V22 source bundle locally.")
    parser.add_argument("--input", required=True, type=Path, help="Raw archive or extracted source directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("appworld_marble",), default="appworld_marble")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source_input = args.input.resolve()
    output = args.output_dir.resolve()
    if not source_input.exists():
        raise FileNotFoundError(source_input)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.name}_working"
    if work.exists():
        raise RuntimeError(f"Scratch directory already exists: {work}")
    work.mkdir(parents=True)
    repo_sft = Path(__file__).resolve().parents[2]
    v19 = repo_sft / "auditor_agent_sft_v19_qualityfix_package"
    v20 = repo_sft / "auditor_agent_sft_v20_appworld_marble_package"
    try:
        if source_input.is_file():
            extracted = work / "extracted"
            extracted.mkdir()
            run("tar", "-xf", source_input, "-C", extracted)
            source_root = find_source_root(extracted)
            source_archive_name = source_input.name
        else:
            source_root = find_source_root(source_input)
            source_archive_name = source_input.name
        assembled = work / "assembled"
        run(
            sys.executable, v20 / "scripts" / "assemble_v20_appworld_marble.py",
            "--source-root", source_root, "--output-dir", assembled, "--seed", args.seed,
            "--source-archive-name", source_archive_name,
            "--source-type", "appworld_marble_v22_source",
        )
        audits = output / "audits"
        audits.mkdir()
        run(
            sys.executable, v19 / "scripts" / "audit_v19_integrity.py",
            "--data-dir", assembled, "--output", audits / "data_integrity.json",
        )
        run(
            sys.executable, v20 / "scripts" / "audit_appworld_observable.py",
            "--data-dir", assembled, "--output", audits / "observable_quality.json",
        )
        base = output / "base_dataset"
        run(
            sys.executable, v20 / "scripts" / "filter_qwen_context_v20.py",
            "--input-dir", assembled, "--output-dir", base,
            "--model", "Qwen/Qwen3-8B", "--revision", QWEN_REVISION,
            "--max-len", 8192,
        )
        run(
            sys.executable, v19 / "scripts" / "audit_lexical_shortcuts.py",
            "--train-file", base / "train.jsonl", "--validation-file", base / "validation.jsonl",
            "--output", audits / "lexical_shortcuts.json",
        )
        train_ids = {
            json.loads(line)["metadata"]["run_id"]
            for line in (base / "train.jsonl").open(encoding="utf-8") if line.strip()
        }
        validation_ids = {
            json.loads(line)["metadata"]["run_id"]
            for line in (base / "validation.jsonl").open(encoding="utf-8") if line.strip()
        }
        if train_ids & validation_ids:
            raise RuntimeError("Train/validation run_id leakage after context filtering")
        manifest = {
            "version": "V22-unified-source-v1",
            "profile": args.profile,
            "source_archive": source_archive_name,
            "seed": args.seed,
            "classification_data": "base_dataset; no Qwen32B enrichment",
            "audit_sft_data": "built on server after ModernBERT validation predictions",
            "train_rows": rows(base / "train.jsonl"),
            "validation_rows": rows(base / "validation.jsonl"),
            "train_sha256": sha256(base / "train.jsonl"),
            "validation_sha256": sha256(base / "validation.jsonl"),
            "train_validation_overlap": 0,
            "sealed_test_accessed": False,
        }
        (output / "SOURCE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        archive = Path(shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name))
        print(json.dumps({"status": "PASS", "bundle": str(output), "archive": str(archive), **manifest}, ensure_ascii=False, indent=2))
    finally:
        resolved_work = work.resolve()
        if resolved_work.parent == output.parent and resolved_work.name == f".{output.name}_working":
            shutil.rmtree(resolved_work, ignore_errors=True)


if __name__ == "__main__":
    main()
