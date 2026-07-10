import argparse
import json
import os
import re
from collections import Counter, defaultdict


AGENT_RE = re.compile(r"agent\d+")


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_content(row, idx):
    return json.loads(row["messages"][idx]["content"])


def agent_idx(agent, nodes):
    try:
        return nodes.index(agent)
    except ValueError:
        return None


def component_to_agents(component_id):
    if not component_id:
        return []
    if component_id.startswith("N::") or component_id.startswith("T::"):
        return AGENT_RE.findall(component_id)
    if component_id.startswith("E::"):
        return AGENT_RE.findall(component_id)
    return []


def gold_agent_indices(user, assistant):
    nodes = user.get("graph", {}).get("nodes", [])
    ids = assistant.get("localization", {}).get("component_ids", []) or []
    indices = set()
    for cid in ids:
        for agent in component_to_agents(cid):
            idx = agent_idx(agent, nodes)
            if idx is not None:
                indices.add(idx)
    return sorted(indices)


def adjacency(user):
    nodes = user.get("graph", {}).get("nodes", [])
    n = len(nodes)
    adj = [[0 for _ in range(n)] for _ in range(n)]
    for edge in user.get("graph", {}).get("edges", []):
        s = agent_idx(edge.get("source"), nodes)
        t = agent_idx(edge.get("target"), nodes)
        if s is not None and t is not None:
            adj[s][t] = 1
    for i in range(n):
        adj[i][i] = 1
    return adj


def event_buckets(user):
    nodes = user.get("graph", {}).get("nodes", [])
    buckets = {agent: [] for agent in nodes}
    ev = user.get("run_evidence", {})
    for event in ev.get("observed", []) + ev.get("final_output", []):
        text = event.get("text", "")
        agents = set()
        for key in ["agent", "source_agent", "target_agent"]:
            if event.get(key):
                agents.add(event[key])
        if not agents:
            for agent in nodes:
                if agent in text:
                    agents.add(agent)
        if not agents:
            # Global summary: expose same run context to every agent, as official
            # code uses system prompts plus utterance embeddings.
            agents = set(nodes)
        for agent in agents:
            if agent in buckets:
                buckets[agent].append(text)
    return buckets


def make_system_prompts(user):
    nodes = user.get("graph", {}).get("nodes", [])
    buckets = event_buckets(user)
    task = user.get("task", {}).get("goal", "")
    prompts = []
    for agent in nodes:
        text = " ".join(buckets.get(agent, [])[:4])
        prompts.append(("Task: " + task + " Agent evidence: " + text)[:1800])
    return prompts


def make_communication_data(user, max_turns=4):
    nodes = user.get("graph", {}).get("nodes", [])
    buckets = event_buckets(user)
    turns = []
    for turn in range(max_turns):
        turn_rows = []
        for idx, agent in enumerate(nodes):
            texts = buckets.get(agent, [])
            txt = texts[turn] if turn < len(texts) else (texts[-1] if texts else f"{agent} has no observed message.")
            turn_rows.append([idx, txt[:1200]])
        turns.append(turn_rows)
    return turns


def convert_row(row):
    user = parse_content(row, 1)
    assistant = parse_content(row, 2)
    metadata = row.get("metadata", {})
    nodes = user.get("graph", {}).get("nodes", [])
    verdict = assistant.get("decision", {}).get("verdict") or metadata.get("verdict")
    attacker_idxes = gold_agent_indices(user, assistant) if verdict == "attack_success" else []
    return {
        "run_id": metadata.get("run_id"),
        "metadata": {
            "scenario": metadata.get("scenario"),
            "topology": metadata.get("topology"),
            "surface": metadata.get("surface"),
            "objective": metadata.get("objective"),
            "verdict": verdict,
            "binary_label": "unsafe" if verdict == "attack_success" else "safe",
        },
        "agents": nodes,
        "adj_matrix": adjacency(user),
        "attacker_idxes": attacker_idxes,
        "system_prompts": make_system_prompts(user),
        "communication_data": make_communication_data(user),
    }


def subset_by_ids(rows, ids_path, allow_missing=True):
    if not ids_path:
        return rows, []
    ids = [line.strip() for line in open(ids_path, encoding="utf-8") if line.strip()]
    by_id = {row["run_id"]: row for row in rows}
    missing = [rid for rid in ids if rid not in by_id]
    if missing and not allow_missing:
        raise SystemExit(f"Missing {len(missing)} ids, first: {missing[:5]}")
    return [by_id[rid] for rid in ids if rid in by_id], missing


def summarize(rows):
    return {
        "total": len(rows),
        "by_verdict": dict(Counter(row["metadata"]["verdict"] for row in rows)),
        "by_surface": dict(Counter(row["metadata"].get("surface") for row in rows)),
        "agent_label_positive_graphs": sum(1 for row in rows if row["attacker_idxes"]),
        "run_unsafe_graphs": sum(1 for row in rows if row["metadata"]["binary_label"] == "unsafe"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data-dir", required=True)
    parser.add_argument("--balanced-ids")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    report = {}
    for split in ["train", "test", "all"]:
        source = load_jsonl(os.path.join(args.sft_data_dir, f"{split}.jsonl"))
        rows = [convert_row(row) for row in source]
        write_jsonl(os.path.join(args.output_dir, f"{split}.jsonl"), rows)
        report[split] = summarize(rows)

    if args.balanced_ids:
        balanced, missing = subset_by_ids([convert_row(row) for row in load_jsonl(os.path.join(args.sft_data_dir, "all.jsonl"))], args.balanced_ids)
        write_jsonl(os.path.join(args.output_dir, "balanced_common.jsonl"), balanced)
        report["balanced_common"] = summarize(balanced)
        report["balanced_common"]["missing_ids"] = missing

    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
