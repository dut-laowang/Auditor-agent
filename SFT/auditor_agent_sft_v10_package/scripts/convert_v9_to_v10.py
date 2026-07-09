import argparse
import json
from pathlib import Path


def parse_json(value):
    return json.loads(value) if isinstance(value, str) else value


def normalize_tools(tools):
    agents = []
    for item in tools or []:
        if isinstance(item, dict) and item.get("agent"):
            agents.append(str(item["agent"]))
        elif isinstance(item, str):
            agents.append(item)
    return sorted(set(agents))


def infer_scope(surface, location_type):
    if surface == "dual" or location_type == "dual":
        return "mixed"
    if surface == "input" or location_type == "global":
        return "global"
    if surface in {"memory", "profile"} or location_type == "node":
        return "node"
    if surface == "message" or location_type == "edge":
        return "edge"
    if surface == "tool" or location_type == "tool_edge":
        return "tool"
    return "none"


def slim_assistant(content):
    old = parse_json(content)
    decision = old.get("decision", {})
    audit = old.get("audit", {}) if isinstance(old.get("audit"), dict) else {}
    inferred = audit.get("inferred_attack", {}) if isinstance(audit.get("inferred_attack"), dict) else {}
    loc = audit.get("localization", {}) if isinstance(audit.get("localization"), dict) else {}
    graph_target = loc.get("graph_target") if isinstance(loc.get("graph_target"), dict) else {}

    surface = inferred.get("surface") or loc.get("surface") or "none"
    objective = inferred.get("objective") or "none"
    scope = graph_target.get("scope") or infer_scope(surface, loc.get("location_type"))
    nodes = graph_target.get("nodes") if graph_target else loc.get("affected_nodes", [])
    edges = graph_target.get("edges") if graph_target else loc.get("affected_edges", [])

    return {
        "decision": {
            "verdict": decision.get("verdict"),
            "binary_label": decision.get("binary_label"),
        },
        "attack": {
            "present": surface != "none",
            "surface": surface,
            "objective": objective,
        },
        "localization": {
            "scope": scope,
            "nodes": sorted(set(str(node) for node in (nodes or []))),
            "edges": [
                {"source": str(edge.get("source")), "target": str(edge.get("target"))}
                for edge in (edges or [])
                if isinstance(edge, dict) and edge.get("source") and edge.get("target")
            ],
            "tools": normalize_tools(graph_target.get("tools", [])),
        },
        "evidence_refs": audit.get("evidence_refs", []),
        "audit_trace": audit.get("audit_trace", []),
    }


def convert_file(input_file: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open(encoding="utf-8") as inp, output_file.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            user = parse_json(row["messages"][1]["content"])
            user["schema"] = "Graph-grounded-Evidence-SFT/v10"
            if str(user.get("sample_uid", "")).startswith("v9_"):
                user["sample_uid"] = "v10_" + user["sample_uid"][3:]
            row["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
            row["messages"][2]["content"] = json.dumps(slim_assistant(row["messages"][2]["content"]), ensure_ascii=False)
            meta = row.get("metadata", {})
            if str(meta.get("sample_uid", "")).startswith("v9_"):
                meta["sample_uid"] = "v10_" + meta["sample_uid"][3:]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-dir", required=True, type=Path)
    parser.add_argument("--v10-dir", required=True, type=Path)
    args = parser.parse_args()
    for name in ["all.jsonl", "train.jsonl", "test.jsonl"]:
        convert_file(args.v9_dir / name, args.v10_dir / name)


if __name__ == "__main__":
    main()
