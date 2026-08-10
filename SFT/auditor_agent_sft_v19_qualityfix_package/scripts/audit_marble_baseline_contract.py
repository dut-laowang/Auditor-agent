"""Dependency-free hard gate for the frozen V19 MARBLE baseline contract."""

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    "train": (4565, "d49ec56577a80fab3be360ae9cb1b90e2d751ce686ffa0f0e2064b2d05d0a932"),
    "validation": (1791, "2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e"),
    "test": (1491, "bee77d962f66f5481e88d89b49b83b3ea9a449e48d776b669ebadd731417167f"),
}
VERDICTS = {"clean_safe", "attack_failed", "attack_success"}
SCOPES = {"none", "global", "node", "edge", "tool", "multi"}
LEAK_PATTERN = re.compile(
    r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|"
    r"success_marker|success_markers|attack_metadata|attack_id|marker_check|"
    r"\[Injected[^\]]*\]|offline verifier|attack-success index|"
    r"labeled as attack-success",
    re.I,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split(path, split):
    rows = 0
    run_ids = set()
    sample_uids = set()
    verdict_counts = {label: 0 for label in sorted(VERDICTS)}
    positive_components = 0
    candidates_total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or [item.get("role") for item in messages] != [
                "system", "user", "assistant"
            ]:
                raise ValueError(f"{split}:{line_number}: invalid message roles")
            if not all(isinstance(item.get("content"), str) for item in messages):
                raise ValueError(f"{split}:{line_number}: non-string message content")
            visible = json.dumps(messages[:2], ensure_ascii=False)
            if LEAK_PATTERN.search(visible):
                raise ValueError(f"{split}:{line_number}: forbidden visible leakage marker")
            user = json.loads(messages[1]["content"])
            target = json.loads(messages[2]["content"])
            metadata = row.get("metadata", {})
            run_id = metadata.get("run_id")
            sample_uid = metadata.get("sample_uid")
            if not run_id or run_id in run_ids:
                raise ValueError(f"{split}:{line_number}: missing or duplicate run_id={run_id}")
            if not sample_uid or sample_uid in sample_uids:
                raise ValueError(f"{split}:{line_number}: missing or duplicate sample_uid={sample_uid}")
            run_ids.add(run_id)
            sample_uids.add(sample_uid)
            verdict = target.get("decision", {}).get("verdict")
            binary = target.get("decision", {}).get("binary_label")
            scope = target.get("localization", {}).get("scope")
            components = target.get("localization", {}).get("component_ids")
            if verdict not in VERDICTS or scope not in SCOPES or not isinstance(components, list):
                raise ValueError(f"{split}:{line_number}: invalid target schema")
            expected_binary = "unsafe" if verdict == "attack_success" else "safe"
            if binary != expected_binary:
                raise ValueError(f"{split}:{line_number}: verdict/binary mismatch")
            if metadata.get("verdict") != verdict:
                raise ValueError(f"{split}:{line_number}: metadata/assistant verdict mismatch")
            candidates = user.get("graph_candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{split}:{line_number}: empty graph_candidates")
            candidate_ids = [
                str(candidate.get("id"))
                for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("id")
            ]
            if len(candidate_ids) != len(candidates) or len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"{split}:{line_number}: invalid or duplicate candidate IDs")
            missing = set(map(str, components)) - set(candidate_ids)
            if missing:
                raise ValueError(f"{split}:{line_number}: gold components absent from candidates: {missing}")
            verdict_counts[verdict] += 1
            positive_components += len(components)
            candidates_total += len(candidate_ids)
    expected_rows, expected_hash = EXPECTED[split]
    actual_hash = sha256(path)
    if rows != expected_rows or actual_hash != expected_hash:
        raise ValueError(
            f"{split}: frozen split mismatch rows={rows}/{expected_rows}, "
            f"sha256={actual_hash}/{expected_hash}"
        )
    return {
        "rows": rows,
        "sha256": actual_hash,
        "run_ids": run_ids,
        "sample_uids": sample_uids,
        "verdict_counts": verdict_counts,
        "positive_components": positive_components,
        "candidates_total": candidates_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    results = {
        split: audit_split(args.data_dir / f"{split}.jsonl", split)
        for split in EXPECTED
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if results[left]["run_ids"] & results[right]["run_ids"]:
            raise ValueError(f"run_id overlap between {left} and {right}")
        if results[left]["sample_uids"] & results[right]["sample_uids"]:
            raise ValueError(f"sample_uid overlap between {left} and {right}")
    serializable = {
        split: {key: value for key, value in result.items() if key not in {"run_ids", "sample_uids"}}
        for split, result in results.items()
    }
    print(json.dumps({"status": "PASS", "contract": "V19 MARBLE", "splits": serializable}, indent=2))


if __name__ == "__main__":
    main()
