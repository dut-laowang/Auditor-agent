import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "server_scripts" / "v19_component_gnn_multitask.py"
SPEC = importlib.util.spec_from_file_location("v19_gnn", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    user = {
        "task": {"scenario": "test", "goal": "audit"},
        "graph": {
            "nodes": ["agent1", "agent2"],
            "edges": [{"source": "agent1", "target": "agent2"}],
        },
        "run_evidence": {
            "observed": [
                {"id": "obs_0", "type": "message_send", "source_agent": "agent1", "target_agent": "agent2", "text": "x"},
                {"id": "obs_unreferenced", "type": "note", "text": "must remain visible"},
            ],
            "final_output": [],
        },
        "graph_candidates": [
            {"id": "G::run", "type": "global", "event_refs": ["obs_0"]},
            {"id": "N::agent1", "type": "node", "agent": "agent1", "local_event_refs": ["obs_0"]},
            {"id": "N::agent2", "type": "node", "agent": "agent2", "incoming_event_refs": ["obs_0"]},
            {"id": "E::agent1->agent2", "type": "edge", "source": "agent1", "target": "agent2", "event_refs": ["obs_0"]},
            {"id": "T::agent1", "type": "tool", "agent": "agent1", "event_refs": []},
        ],
    }
    edge_index = MODULE.component_edges(user, user["graph_candidates"])
    edges = {tuple(edge) for edge in edge_index.T.tolist()}
    assert (0, 4) in edges and (4, 0) in edges
    assert (1, 2) in edges and (2, 1) in edges
    assert (3, 1) in edges and (3, 2) in edges
    assert MODULE.predicted_scope([]) == "none"
    assert MODULE.predicted_scope(["N::agent1", "N::agent2"]) == "node"
    assert MODULE.predicted_scope(["N::agent1", "T::agent1"]) == "multi"
    pieces = MODULE.candidate_pieces(user, user["graph_candidates"][0], MODULE.event_index(user))
    assert any("must remain visible" in piece for piece in pieces)
    records = [
        {
            "gold": "attack_success",
            "pred": "attack_success",
            "gold_scope": "node",
            "pred_scope": "node",
            "gold_components": ["N::agent1"],
            "pred_components": ["N::agent1"],
        }
    ]
    result = MODULE.localization_summary(records, lambda row: True)
    assert result["component_micro_f1"] == 1.0
    assert result["component_exact_match"] == 1.0
    print({"status": "PASS", "tests": 8})


if __name__ == "__main__":
    main()
