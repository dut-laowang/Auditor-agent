import argparse
import json
import os
from collections import Counter


TYPE_ORDER = ["global", "node", "edge", "tool"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_message(row, index):
    return json.loads(row["messages"][index]["content"])


def event_index(user):
    events = {}
    ev = user.get("run_evidence", {})
    groups = [
        ev.get("observed", []),
        ev.get("final_output", []),
        ev.get("reference", {}).get("clean", []),
    ]
    for group in groups:
        for event in group:
            if isinstance(event, dict) and event.get("id"):
                events[event["id"]] = event
    return events


def candidate_refs(candidate):
    refs = []
    for key in ["event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"]:
        refs.extend(str(ref) for ref in candidate.get(key, []) or [])
    return refs


def truncate(text, limit=1400):
    text = " ".join(str(text or "").split())
    return text[:limit]


def candidate_text(candidate, events):
    refs = candidate_refs(candidate)
    pieces = []
    for ref in refs[:6]:
        event = events.get(ref)
        if not event:
            continue
        etype = event.get("type", "event")
        agent = event.get("agent") or event.get("source_agent") or ""
        target = event.get("target_agent") or ""
        prefix = f"{etype} {agent}->{target}".strip()
        pieces.append(f"{prefix}: {event.get('text', '')}")
    if not pieces:
        desc = candidate.get("description") or candidate.get("id") or candidate.get("type")
        pieces.append(f"structural candidate: {desc}")
    return truncate(" ".join(pieces))


def relation_text(src, dst, candidates):
    a = candidates[src]
    b = candidates[dst]
    return truncate(f"{a.get('id')} interacts with {b.get('id')} through MAS topology/evidence relation.", 400)


def connect_candidates(candidates):
    by_id = {c.get("id"): idx for idx, c in enumerate(candidates)}
    edges = set()
    global_idx = by_id.get("G::run")
    for idx, cand in enumerate(candidates):
        if global_idx is not None and idx != global_idx:
            edges.add((global_idx, idx))
            edges.add((idx, global_idx))
        if cand.get("type") == "edge":
            src = cand.get("source")
            tgt = cand.get("target")
            for node_id in [f"N::{src}", f"N::{tgt}"]:
                if node_id in by_id:
                    edges.add((idx, by_id[node_id]))
                    edges.add((by_id[node_id], idx))
        if cand.get("type") == "tool":
            node_id = f"N::{cand.get('agent')}"
            if node_id in by_id:
                edges.add((idx, by_id[node_id]))
                edges.add((by_id[node_id], idx))
    for i in range(len(candidates)):
        edges.add((i, i))
    return sorted(edges)


def type_features(candidate):
    return [1.0 if candidate.get("type") == name else 0.0 for name in TYPE_ORDER]


def build_row(row):
    user = parse_message(row, 1)
    assistant = parse_message(row, 2)
    metadata = row.get("metadata", {})
    candidates = user.get("graph_candidates", [])
    events = event_index(user)
    loc_ids = set(assistant.get("localization", {}).get("component_ids", []) or [])
    decision = assistant.get("decision", {})
    attack = assistant.get("attack", {})
    edge_index = connect_candidates(candidates)
    return {
        "run_id": metadata.get("run_id"),
        "sample_uid": metadata.get("sample_uid") or user.get("sample_uid"),
        "schema": "MAS-GNN-Strong-Graph/v1",
        "metadata": {
            "scenario": metadata.get("scenario"),
            "topology": metadata.get("topology"),
            "surface": attack.get("surface") or metadata.get("surface"),
            "objective": attack.get("objective") or metadata.get("objective"),
            "verdict": decision.get("verdict") or metadata.get("verdict"),
            "binary_label": decision.get("binary_label") or metadata.get("label"),
        },
        "candidate_ids": [candidate.get("id") for candidate in candidates],
        "candidate_types": [candidate.get("type") for candidate in candidates],
        "candidate_texts": [candidate_text(candidate, events) for candidate in candidates],
        "candidate_type_features": [type_features(candidate) for candidate in candidates],
        "edge_index": edge_index,
        "edge_texts": [relation_text(src, dst, candidates) for src, dst in edge_index],
        "component_labels": [1 if candidate.get("id") in loc_ids else 0 for candidate in candidates],
        "run_label": 1 if decision.get("verdict") == "attack_success" else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for split in ["train", "test", "all"]:
        rows = [build_row(row) for row in load_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"))]
        dump_jsonl(os.path.join(args.output_dir, f"{split}.jsonl"), rows)
        summary[split] = {
            "total": len(rows),
            "run_labels": dict(Counter(row["run_label"] for row in rows)),
            "component_positive": sum(sum(row["component_labels"]) for row in rows),
            "avg_candidates": round(sum(len(row["candidate_ids"]) for row in rows) / max(len(rows), 1), 3),
            "avg_edges": round(sum(len(row["edge_index"]) for row in rows) / max(len(rows), 1), 3),
            "by_surface": dict(Counter(row["metadata"].get("surface") for row in rows)),
        }
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
