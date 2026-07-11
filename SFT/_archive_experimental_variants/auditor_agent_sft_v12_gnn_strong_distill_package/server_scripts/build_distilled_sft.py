import argparse
import json
import os
import re
from collections import Counter


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


def score_map(score_rows):
    return {row["run_id"]: row for row in score_rows}


def risk_bucket(score):
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def parse(row, idx):
    return json.loads(row["messages"][idx]["content"])


def candidate_ids(user):
    return {candidate.get("id") for candidate in user.get("graph_candidates", [])}


def valid_refs(user):
    refs = candidate_ids(user)
    ev = user.get("run_evidence", {})
    for group in [ev.get("observed", []), ev.get("final_output", []), ev.get("reference", {}).get("clean", [])]:
        for event in group:
            if isinstance(event, dict) and event.get("id"):
                refs.add(event["id"])
    return refs


def enrich_user(user, scores):
    user = dict(user)
    user["schema"] = "Graph-grounded-GNN-Distill-SFT/v1"
    return user


def enrich_assistant(assistant, scores):
    assistant = dict(assistant)
    loc = dict(assistant.get("localization", {}))
    gold_ids = set(loc.get("component_ids", []) or [])
    ranking = []
    for item in scores.get("component_ranking", [])[:8]:
        entry = {"id": item["id"], "risk": risk_bucket(float(item["score"])), "type": item.get("type")}
        if item["id"] in gold_ids:
            entry["target"] = True
        ranking.append(entry)
    loc["candidate_ranking"] = ranking
    assistant["localization"] = loc
    trace = assistant.get("audit_trace", [])
    if isinstance(trace, list):
        for step in trace:
            if isinstance(step, dict) and step.get("step") == "localize_component":
                step["component_refs"] = loc.get("component_ids", []) or step.get("component_refs", [])
    assistant["audit_trace"] = trace
    return assistant


def check_row(row):
    text = row["messages"][1]["content"]
    hits = []
    for pat in LEAK_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    user = parse(row, 1)
    assistant = parse(row, 2)
    refs = valid_refs(user)
    invalid = []
    for step in assistant.get("audit_trace", []) or []:
        if not isinstance(step, dict):
            continue
        for ref in step.get("evidence_refs", []) or []:
            if ref not in refs:
                invalid.append(ref)
        for ref in step.get("component_refs", []) or []:
            if ref not in refs:
                invalid.append(ref)
    return hits, invalid


def convert_split(seed_dir, score_index, split):
    out_rows = []
    missing = 0
    for row in load_jsonl(os.path.join(seed_dir, f"{split}.jsonl")):
        rid = row.get("metadata", {}).get("run_id")
        scores = score_index.get(rid)
        if scores is None:
            missing += 1
            scores = {"component_ranking": []}
        user = enrich_user(parse(row, 1), scores)
        assistant = enrich_assistant(parse(row, 2), scores)
        new_row = {
            "messages": [
                row["messages"][0],
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
            ],
            "metadata": dict(row.get("metadata", {})),
        }
        new_row["metadata"]["schema"] = "Graph-grounded-GNN-Distill-SFT/v1"
        new_row["metadata"]["distill_source"] = "light_gnn_component_ranking"
        out_rows.append(new_row)
    return out_rows, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-data-dir", required=True)
    parser.add_argument("--gnn-score-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scores = score_map(load_jsonl(args.gnn_score_file))
    stats = {"files": {}, "leak_check": {}, "invalid_trace_refs": {}}
    for split in ["all", "train", "test"]:
        rows, missing = convert_split(args.seed_data_dir, scores, split)
        write_jsonl(os.path.join(args.output_dir, f"{split}.jsonl"), rows)
        verdicts = Counter(row["metadata"].get("verdict") for row in rows)
        leak_hits = Counter()
        invalid_refs = 0
        for row in rows:
            hits, invalid = check_row(row)
            leak_hits.update(hits)
            invalid_refs += len(invalid)
        stats["files"][split] = {"total": len(rows), "missing_gnn_scores": missing, "by_verdict": dict(verdicts)}
        stats["leak_check"][split] = dict(leak_hits)
        stats["invalid_trace_refs"][split] = invalid_refs
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
