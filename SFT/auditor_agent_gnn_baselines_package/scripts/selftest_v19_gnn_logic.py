import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "server_scripts" / "v19_component_gnn_multitask.py"
SPEC = importlib.util.spec_from_file_location("v19_gnn", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
UNSUPERVISED_SCRIPT = Path(__file__).resolve().parents[1] / "server_scripts" / "v19_unsupervised_graph_baselines.py"
UNSUPERVISED_SPEC = importlib.util.spec_from_file_location("v19_unsupervised", UNSUPERVISED_SCRIPT)
UNSUPERVISED = importlib.util.module_from_spec(UNSUPERVISED_SPEC)
UNSUPERVISED_SPEC.loader.exec_module(UNSUPERVISED)


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
    assert result["by_gold_scope"]["node"]["scope_accuracy"] == 1.0
    tensor_row = {
        "x": np.zeros((1, 389), dtype=np.float32),
        "edge_index": np.asarray([[0], [0]], dtype=np.int64),
        "gold_verdict": "attack_success",
        "gold_scope": "node",
        "gold_components": ["N::agent1"],
        "candidate_ids": ["N::agent1"],
    }
    _, _, verdict, scope, components = MODULE.graph_tensors(tensor_row, torch.device("cpu"))
    assert verdict.item() == MODULE.VERDICT_TO_ID["attack_success"]
    assert scope.item() == MODULE.SCOPE_TO_ID["node"]
    assert components.tolist() == [1.0]
    assert UNSUPERVISED.verdict_from_score(0.1, 0.2, 0.8) == "clean_safe"
    assert UNSUPERVISED.verdict_from_score(0.5, 0.2, 0.8) == "attack_failed"
    assert UNSUPERVISED.verdict_from_score(0.9, 0.2, 0.8) == "attack_success"
    fused = UNSUPERVISED.XGGuardModel.fuse(torch.tensor([1.0, 2.0, 4.0]), torch.tensor([4.0, 2.0, 1.0]))
    assert fused.shape == (3,) and torch.isfinite(fused).all()
    restored_states = MODULE.cpu_cuda_rng_states([torch.arange(8, dtype=torch.uint8)])
    assert restored_states[0].device.type == "cpu" and restored_states[0].dtype == torch.uint8
    print({"status": "PASS", "tests": 17})


if __name__ == "__main__":
    main()
