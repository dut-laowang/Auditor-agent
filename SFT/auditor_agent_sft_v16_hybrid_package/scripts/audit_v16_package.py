from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    malformed: list[dict] = []
    for mode in ("single", "dual"):
        merged = args.source_root / mode / "final" / "merged"
        for kind in ("trajectories", "marble_results"):
            for path in sorted((merged / kind).glob("*.jsonl")):
                bad = []
                with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            bad.append(number)
                if bad:
                    malformed.append(
                        {
                            "mode": mode,
                            "kind": kind,
                            "file": path.name,
                            "bad_line_count": len(bad),
                            "bad_line_numbers": bad,
                        }
                    )

    qc_path = args.dataset_dir / "manual_quality_sample_50_v16.json"
    qc_document = json.loads(qc_path.read_text(encoding="utf-8"))
    qc = qc_document["samples"]
    checks = Counter()
    strata = Counter()
    failures = []
    expected = {
        "success": ("attack_success", "unsafe"),
        "failure": ("attack_failed", "safe"),
        "private_control": ("clean_safe", "safe"),
        "clean": ("clean_safe", "safe"),
    }
    for i, sample in enumerate(qc):
        meta = sample["metadata"]
        user = json.loads(sample["messages"][1]["content"])
        target = json.loads(sample["messages"][2]["content"])
        source_label = meta["source_final_label"]
        exp_verdict, exp_binary = expected[source_label]
        checks["rows"] += 1
        checks["valid_message_triplet"] += len(sample["messages"]) == 3
        checks["source_mapping_ok"] += (
            target.get("decision", {}).get("verdict") == exp_verdict
            and target.get("decision", {}).get("binary_label") == exp_binary
        )
        checks["has_visible_evidence"] += bool(
            user.get("run_evidence", {}).get("observed")
            or user.get("run_evidence", {}).get("final_output")
        )
        checks["has_candidates"] += bool(user.get("graph_candidates"))
        checks["assistant_json_object"] += isinstance(target, dict)
        strata[
            (
                meta.get("scenario"),
                meta.get("attack_mode"),
                target.get("decision", {}).get("verdict"),
            )
        ] += 1
        if not (
            len(sample["messages"]) == 3
            and target.get("decision", {}).get("verdict") == exp_verdict
            and target.get("decision", {}).get("binary_label") == exp_binary
        ):
            failures.append({"index": i, "id": sample.get("id")})

    hashes = {}
    line_counts = {}
    for name in ("all.jsonl", "train.jsonl", "test.jsonl"):
        path = args.dataset_dir / name
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                count += bool(line.strip())
        hashes[name] = digest.hexdigest()
        line_counts[name] = count

    report = {
        "quality_control_50": {
            "checks": dict(checks),
            "failures": failures,
            "strata": [
                {
                    "scenario": key[0],
                    "attack_mode": key[1],
                    "verdict": key[2],
                    "count": value,
                }
                for key, value in sorted(strata.items())
            ],
        },
        "source_malformed_jsonl": {
            "affected_files": len(malformed),
            "bad_lines": sum(x["bad_line_count"] for x in malformed),
            "by_mode": dict(Counter(
                {mode: sum(x["bad_line_count"] for x in malformed if x["mode"] == mode)
                 for mode in ("single", "dual")}
            )),
            "by_kind": dict(Counter(
                {kind: sum(x["bad_line_count"] for x in malformed if x["kind"] == kind)
                 for kind in ("trajectories", "marble_results")}
            )),
            "files": malformed,
        },
        "dataset_line_counts": line_counts,
        "sha256": hashes,
    }
    (args.dataset_dir / "quality_audit_v16.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
