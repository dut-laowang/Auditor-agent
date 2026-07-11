import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


LEAK_PATTERNS = [
    re.compile(r"ACI_[A-Za-z0-9_]+"),
    re.compile(r"attack_id", re.I),
    re.compile(r"success_markers?", re.I),
    re.compile(r"attack_metadata", re.I),
    re.compile(r"\[Injected instruction\]", re.I),
]


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse(row, idx):
    return json.loads(row["messages"][idx]["content"])


def candidate_refs(candidate):
    refs = []
    for key in ["event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"]:
        refs.extend(str(ref) for ref in candidate.get(key, []) or [])
    return refs


def evidence_stats(user, assistant):
    ev = user.get("run_evidence", {})
    obs = ev.get("observed", [])
    final = ev.get("final_output", [])
    refs = ev.get("reference", {}).get("clean", [])
    candidates = {candidate.get("id"): candidate for candidate in user.get("graph_candidates", [])}
    loc = assistant.get("localization", {})
    component_ids = loc.get("component_ids", []) or []
    comp_ref_count = 0
    for cid in component_ids:
        comp_ref_count += len(candidate_refs(candidates.get(cid, {})))
    observed_types = Counter(event.get("type") for event in obs + final + refs if isinstance(event, dict))
    text_len = sum(len(event.get("text", "")) for event in obs + final if isinstance(event, dict))
    return {
        "observed": len(obs),
        "final": len(final),
        "clean_refs": len(refs),
        "component_refs": comp_ref_count,
        "text_len": text_len,
        "event_types": dict(observed_types),
    }


def should_keep(row, user, assistant):
    metadata = row.get("metadata", {})
    verdict = metadata.get("verdict")
    stats = evidence_stats(user, assistant)
    # Keep all clean rows unless they are completely empty.
    if verdict == "clean_safe":
        return stats["observed"] >= 1 and stats["text_len"] >= 250
    # Remove only very sparse attacked rows. Most V12 rows have enough run-level evidence.
    if stats["final"] < 1:
        return False
    if stats["observed"] < 2:
        return False
    if stats["text_len"] < 500:
        return False
    return True


def localization_confidence(row, user, assistant):
    metadata = row.get("metadata", {})
    surface = metadata.get("surface")
    verdict = metadata.get("verdict")
    loc = assistant.get("localization", {})
    scope = loc.get("scope")
    component_ids = loc.get("component_ids", []) or []
    stats = evidence_stats(user, assistant)
    if verdict == "clean_safe" or scope == "none":
        return "high", "none"
    if not component_ids:
        return "low", "no_component"
    if scope == "global" or surface == "input":
        return "high", "global_input"
    if scope == "mixed" or surface == "dual":
        if stats["component_refs"] >= 2:
            return "medium", "mixed_with_refs"
        return "low", "mixed_low_refs"
    if stats["component_refs"] >= 2:
        return "high", "component_has_multiple_refs"
    if stats["component_refs"] == 1:
        return "medium", "component_has_single_ref"
    return "low", "component_without_refs"


def downgrade_localization(assistant, confidence, reason):
    out = dict(assistant)
    loc = dict(out.get("localization", {}))
    loc["confidence"] = confidence
    loc["quality_reason"] = reason
    loc["localization_required"] = confidence in {"high", "medium"}
    if confidence == "low":
        loc["scope"] = "unknown"
        loc["component_ids"] = []
    out["localization"] = loc
    trace = []
    for step in out.get("audit_trace", []) or []:
        if not isinstance(step, dict):
            continue
        new_step = dict(step)
        if confidence == "low":
            new_step["component_refs"] = []
            if new_step.get("step") == "localize_component":
                new_step["claim"] = "Component-level localization is not emitted because observed evidence is insufficient."
        trace.append(new_step)
    out["audit_trace"] = trace
    return out


def valid_refs(user):
    refs = set()
    ev = user.get("run_evidence", {})
    for group in [ev.get("observed", []), ev.get("final_output", []), ev.get("reference", {}).get("clean", [])]:
        for event in group:
            if isinstance(event, dict) and event.get("id"):
                refs.add(event["id"])
    for candidate in user.get("graph_candidates", []):
        if candidate.get("id"):
            refs.add(candidate["id"])
        refs.update(candidate_refs(candidate))
    return refs


def trace_invalid_refs(user, assistant):
    refs = valid_refs(user)
    invalid = []
    for step in assistant.get("audit_trace", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ["evidence_refs", "component_refs"]:
            for ref in step.get(key, []) or []:
                if ref not in refs:
                    invalid.append(ref)
    return invalid


def leak_hits(row):
    text = row["messages"][1]["content"]
    hits = []
    for pat in LEAK_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def convert_rows(rows):
    out = []
    report = {
        "source_rows": len(rows),
        "kept_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": Counter(),
        "confidence": Counter(),
        "quality_reason": Counter(),
        "by_verdict": Counter(),
        "by_surface": Counter(),
        "invalid_trace_refs": 0,
        "leak_hits": Counter(),
    }
    for row in rows:
        user = parse(row, 1)
        assistant = parse(row, 2)
        metadata = dict(row.get("metadata", {}))
        if not should_keep(row, user, assistant):
            report["dropped_rows"] += 1
            report["drop_reasons"]["sparse_run_evidence"] += 1
            continue
        confidence, reason = localization_confidence(row, user, assistant)
        assistant = downgrade_localization(assistant, confidence, reason)
        user = dict(user)
        user["schema"] = "Graph-grounded-Candidate-SFT/v12-hq-final"
        metadata["schema"] = "Graph-grounded-Candidate-SFT/v12-hq-final"
        metadata["hq_localization_confidence"] = confidence
        metadata["hq_quality_reason"] = reason
        new_row = {
            "messages": [
                row["messages"][0],
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
            ],
            "metadata": metadata,
        }
        report["kept_rows"] += 1
        report["confidence"][confidence] += 1
        report["quality_reason"][reason] += 1
        report["by_verdict"][metadata.get("verdict")] += 1
        report["by_surface"][metadata.get("surface")] += 1
        report["invalid_trace_refs"] += len(trace_invalid_refs(user, assistant))
        report["leak_hits"].update(leak_hits(new_row))
        out.append(new_row)
    return out, report


def split_like_source(source_train, source_test, hq_all_by_id):
    train = []
    test = []
    for row in source_train:
        rid = row.get("metadata", {}).get("run_id")
        if rid in hq_all_by_id:
            train.append(hq_all_by_id[rid])
    for row in source_test:
        rid = row.get("metadata", {}).get("run_id")
        if rid in hq_all_by_id:
            test.append(hq_all_by_id[rid])
    return train, test


def summarize(rows):
    return {
        "total": len(rows),
        "by_verdict": dict(Counter(row["metadata"].get("verdict") for row in rows)),
        "by_surface": dict(Counter(row["metadata"].get("surface") for row in rows)),
        "by_confidence": dict(Counter(row["metadata"].get("hq_localization_confidence") for row in rows)),
        "by_quality_reason": dict(Counter(row["metadata"].get("hq_quality_reason") for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_all = load_jsonl(os.path.join(args.v12_data_dir, "all.jsonl"))
    source_train = load_jsonl(os.path.join(args.v12_data_dir, "train.jsonl"))
    source_test = load_jsonl(os.path.join(args.v12_data_dir, "test.jsonl"))
    hq_all, report = convert_rows(source_all)
    by_id = {row["metadata"]["run_id"]: row for row in hq_all}
    hq_train, hq_test = split_like_source(source_train, source_test, by_id)
    write_jsonl(os.path.join(args.output_dir, "all.jsonl"), hq_all)
    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), hq_train)
    write_jsonl(os.path.join(args.output_dir, "test.jsonl"), hq_test)
    random.seed(args.seed)
    sample = random.sample(hq_all, min(args.sample_size, len(hq_all)))
    with open(os.path.join(args.output_dir, "manual_quality_sample_50_hq.json"), "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)
    stats = {
        "schema": "Graph-grounded-Candidate-SFT/v12-hq-final",
        "policy": "Preserve V12 run-level labels and split; filter only sparse rows; downgrade low-evidence component localization instead of forcing noisy component labels.",
        "source": {"all": len(source_all), "train": len(source_train), "test": len(source_test)},
        "files": {"all": summarize(hq_all), "train": summarize(hq_train), "test": summarize(hq_test)},
        "conversion_report": {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in report.items()
        },
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
