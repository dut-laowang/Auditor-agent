import argparse
import json
import os
from collections import Counter, defaultdict


TYPE_ORDER = ["global", "node", "edge", "tool"]
SURFACE_ORDER = ["none", "input", "memory", "message", "profile", "tool", "dual"]
OBJECTIVE_ORDER = ["none", "disclosure", "disruption", "hijacking"]


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


def all_refs(user):
    refs = {}
    ev = user.get("run_evidence", {})
    for group in [ev.get("observed", []), ev.get("final_output", []), ev.get("reference", {}).get("clean", [])]:
        for event in group:
            if isinstance(event, dict) and event.get("id"):
                refs[event["id"]] = event
    return refs


def candidate_refs(candidate):
    refs = []
    for key in ["event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"]:
        refs.extend(str(ref) for ref in candidate.get(key, []) or [])
    return refs


def agent_for_candidate(candidate):
    if candidate.get("agent"):
        return candidate["agent"]
    if candidate.get("source"):
        return candidate["source"]
    cid = candidate.get("id", "")
    if cid.startswith("N::") or cid.startswith("T::"):
        return cid.split("::", 1)[1]
    return ""


def candidate_features(candidate, user, ref_index, graph):
    refs = candidate_refs(candidate)
    ref_events = [ref_index[r] for r in refs if r in ref_index]
    type_vec = [1.0 if candidate.get("type") == t else 0.0 for t in TYPE_ORDER]
    event_type_counts = Counter(e.get("type", "") for e in ref_events)
    text_len = sum(len(e.get("text", "")) for e in ref_events)
    agent = agent_for_candidate(candidate)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    in_degree = sum(1 for edge in edges if edge.get("target") == agent)
    out_degree = sum(1 for edge in edges if edge.get("source") == agent)
    return type_vec + [
        min(len(refs), 8) / 8.0,
        min(event_type_counts.get("agent_act", 0), 4) / 4.0,
        min(event_type_counts.get("message_send", 0), 4) / 4.0,
        min(event_type_counts.get("message_receive", 0), 4) / 4.0,
        min(event_type_counts.get("tool_call", 0) + event_type_counts.get("tool_result", 0), 4) / 4.0,
        min(event_type_counts.get("run_summary", 0), 4) / 4.0,
        min(text_len, 1200) / 1200.0,
        min(in_degree, 8) / 8.0,
        min(out_degree, 8) / 8.0,
        min(len(nodes), 12) / 12.0,
    ]


def connect_candidates(candidates):
    by_id = {c.get("id"): idx for idx, c in enumerate(candidates)}
    edges = set()
    global_idx = by_id.get("G::run")
    for idx, cand in enumerate(candidates):
        if global_idx is not None and idx != global_idx:
            edges.add((global_idx, idx))
            edges.add((idx, global_idx))
        ctype = cand.get("type")
        if ctype == "edge":
            src = cand.get("source")
            tgt = cand.get("target")
            for node_id in [f"N::{src}", f"N::{tgt}"]:
                if node_id in by_id:
                    edges.add((idx, by_id[node_id]))
                    edges.add((by_id[node_id], idx))
        elif ctype == "tool":
            agent = cand.get("agent")
            node_id = f"N::{agent}"
            if node_id in by_id:
                edges.add((idx, by_id[node_id]))
                edges.add((by_id[node_id], idx))
    for i in range(len(candidates)):
        edges.add((i, i))
    return sorted(edges)


def build_row(row):
    user = parse_message(row, 1)
    assistant = parse_message(row, 2)
    metadata = row.get("metadata", {})
    candidates = user.get("graph_candidates", [])
    ref_index = all_refs(user)
    graph = user.get("graph", {})
    loc_ids = set(assistant.get("localization", {}).get("component_ids", []) or [])
    decision = assistant.get("decision", {})
    attack = assistant.get("attack", {})
    labels = [1 if candidate.get("id") in loc_ids else 0 for candidate in candidates]
    return {
        "run_id": metadata.get("run_id"),
        "sample_uid": metadata.get("sample_uid") or user.get("sample_uid"),
        "schema": "MAS-GNN-Distill-Graph/v1",
        "split_hint": metadata.get("split"),
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
        "features": [candidate_features(candidate, user, ref_index, graph) for candidate in candidates],
        "edge_index": connect_candidates(candidates),
        "component_labels": labels,
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
            "by_surface": dict(Counter(row["metadata"].get("surface") for row in rows)),
        }
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
