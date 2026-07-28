from __future__ import annotations

import argparse
import hashlib
import json
import re
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

    qc_path = args.dataset_dir / "manual_quality_sample_50_v16_1.json"
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
    split_prompts = {}
    split_groups = {}
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
        if name in {"train.jsonl", "test.jsonl"}:
            prompts = set()
            normalized_prompts = set()
            groups = set()
            privileged_hits = Counter()
            rule_correct = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    text = row["messages"][1]["content"]
                    user = json.loads(text)
                    gold = json.loads(row["messages"][2]["content"])["decision"]["verdict"]
                    prompts.add(hashlib.sha256(text.encode()).hexdigest())
                    normalized = re.sub(
                        r"v16_1_\d+|(?:obs|out|ref)_\d+|"
                        r"[0-9a-f]{8}-[0-9a-f-]{27,}",
                        "<ID>",
                        text,
                        flags=re.I,
                    )
                    normalized_prompts.add(
                        hashlib.sha256(normalized.encode()).hexdigest()
                    )
                    meta = row["metadata"]
                    groups.add((meta["scenario"], int(meta["sample_id"])))
                    sources = {
                        event.get("source")
                        for section in ("observed", "final_output")
                        for event in user["run_evidence"].get(section, [])
                    }
                    if "hq_visible_output_marker_scan" in sources:
                        privileged_hits["marker_scan_source"] += 1
                    if user["run_evidence"]["reference"].get("available"):
                        privileged_hits["reference_available"] += 1
                    if user["run_evidence"]["coverage"].get("has_clean_contrast"):
                        privileged_hits["has_clean_contrast"] += 1
                    pred = (
                        "attack_success"
                        if "hq_visible_output_marker_scan" in sources
                        else (
                            "clean_safe"
                            if user["run_evidence"]["reference"].get("available")
                            else "attack_failed"
                        )
                    )
                    rule_correct += pred == gold
            split_prompts[name] = {
                "exact": prompts,
                "normalized": normalized_prompts,
                "privileged_hits": dict(privileged_hits),
                "legacy_proxy_rule_accuracy": rule_correct / line_counts[name],
            }
            split_groups[name] = groups

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
        "split_leakage": {
            "exact_prompt_overlap": len(
                split_prompts["train.jsonl"]["exact"]
                & split_prompts["test.jsonl"]["exact"]
            ),
            "normalized_prompt_overlap": len(
                split_prompts["train.jsonl"]["normalized"]
                & split_prompts["test.jsonl"]["normalized"]
            ),
            "task_group_overlap": len(
                split_groups["train.jsonl"] & split_groups["test.jsonl"]
            ),
            "train_privileged_hits": split_prompts["train.jsonl"]["privileged_hits"],
            "test_privileged_hits": split_prompts["test.jsonl"]["privileged_hits"],
            "legacy_proxy_rule_accuracy_train": split_prompts["train.jsonl"][
                "legacy_proxy_rule_accuracy"
            ],
            "legacy_proxy_rule_accuracy_test": split_prompts["test.jsonl"][
                "legacy_proxy_rule_accuracy"
            ],
        },
        "sha256": hashes,
    }
    (args.dataset_dir / "quality_audit_v16_1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
