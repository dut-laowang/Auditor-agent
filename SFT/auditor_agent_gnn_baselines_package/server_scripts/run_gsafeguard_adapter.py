import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def ensure_import_official(path):
    sys.path.insert(0, str(Path(path).resolve()))
    from model import MyGAT  # noqa: import-error
    return MyGAT


def encode_rows(rows, model_name):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    encoded = []
    for row in tqdm(rows, desc="encode_gsafeguard_graphs"):
        features = encoder.encode(row["system_prompts"], normalize_embeddings=True)
        comm = []
        for turn in row["communication_data"]:
            texts = [text for _, text in sorted(turn, key=lambda x: x[0])]
            comm.append(encoder.encode(texts, normalize_embeddings=True))
        encoded.append({**row, "features": np.asarray(features, dtype=np.float32), "turn_embeddings": np.asarray(comm, dtype=np.float32)})
    return encoded


def to_graph(row, device):
    adj = np.asarray(row["adj_matrix"])
    edge_index_np = np.array(adj.nonzero(), dtype=np.int64)
    if edge_index_np.size == 0:
        edge_index_np = np.array([[0], [0]], dtype=np.int64)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    turn_embeddings = torch.tensor(row["turn_embeddings"], dtype=torch.float32, device=device)  # [T,N,D]
    # Official gen_training_dataset uses communication_embeddings[edge_index[1]]
    edge_attr = turn_embeddings.permute(1, 0, 2)[edge_index[1]]  # [E,T,D]
    x = torch.tensor(row["features"], dtype=torch.float32, device=device)
    labels = torch.zeros(len(row["agents"]), dtype=torch.float32, device=device)
    for idx in row["attacker_idxes"]:
        if 0 <= idx < len(labels):
            labels[idx] = 1.0
    return x, edge_index, edge_attr, labels


def train_model(MyGAT, train_rows, device, epochs, lr, hidden_dim):
    x0, edge_index0, edge_attr0, y0 = to_graph(train_rows[0], device)
    model = MyGAT(
        in_channels=x0.size(-1),
        hidden_channels=hidden_dim,
        out_channels=1,
        heads=8,
        num_layers=2,
        edge_dim=edge_attr0.size()[1:],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    pos_weight = torch.tensor(4.0, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for row in tqdm(train_rows, desc=f"gsafeguard_epoch_{epoch}"):
            x, edge_index, edge_attr, labels = to_graph(row, device)
            logits = model(x, edge_index, edge_attr).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        print(json.dumps({"epoch": epoch, "loss": total / max(len(train_rows), 1)}))
    return model


def evaluate(model, rows, device, output_path, agent_label_policy):
    records = []
    gold_run, pred_run, score_run = [], [], []
    gold_agent, pred_agent = [], []
    top1_hits = []
    top2_hits = []
    model.eval()
    with torch.no_grad():
        for row in tqdm(rows, desc="eval_gsafeguard"):
            x, edge_index, edge_attr, labels = to_graph(row, device)
            scores = torch.sigmoid(model(x, edge_index, edge_attr).squeeze(-1)).detach().cpu().tolist()
            run_score = max(scores) if scores else 0.0
            run_pred = 1 if run_score >= 0.5 else 0
            run_gold = 1 if row["metadata"]["binary_label"] == "unsafe" else 0
            ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            gold_set = set(row["attacker_idxes"])
            pred_set = {i for i, s in enumerate(scores) if s >= 0.5}
            gold_run.append(run_gold)
            pred_run.append(run_pred)
            score_run.append(run_score)
            for i in range(len(scores)):
                gold_agent.append(1 if i in gold_set else 0)
                pred_agent.append(1 if i in pred_set else 0)
            if gold_set:
                top1_hits.append(1 if ranking and ranking[0] in gold_set else 0)
                top2_hits.append(1 if any(i in gold_set for i in ranking[:2]) else 0)
            records.append({
                "run_id": row["run_id"],
                "gold_binary": row["metadata"]["binary_label"],
                "pred_binary": "unsafe" if run_pred else "safe",
                "run_score": run_score,
                "gold_agents": [row["agents"][i] for i in row["attacker_idxes"] if i < len(row["agents"])],
                "ranked_agents": [{"agent": row["agents"][i], "score": scores[i]} for i in ranking],
            })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        "method": "G-Safeguard-style supervised GAT adapter",
        "agent_label_policy": agent_label_policy,
        "n": len(rows),
        "comparable_metrics": {
            "binary_accuracy": accuracy_score(gold_run, pred_run),
            "binary_report": classification_report(gold_run, pred_run, labels=[0, 1], target_names=["safe", "unsafe"], zero_division=0, output_dict=True),
            "agent_localization_report": classification_report(gold_agent, pred_agent, labels=[0, 1], target_names=["not_target", "target_agent"], zero_division=0, output_dict=True),
            "agent_top1_hit": float(np.mean(top1_hits)) if top1_hits else None,
            "agent_top2_hit": float(np.mean(top2_hits)) if top2_hits else None,
        },
        "not_applicable": ["surface_f1", "objective_f1", "edge_tool_global_localization", "json_audit_quality"],
    }
    try:
        metrics["comparable_metrics"]["run_auroc"] = roc_auc_score(gold_run, score_run)
    except Exception:
        metrics["comparable_metrics"]["run_auroc"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-ta-dir", required=True)
    parser.add_argument("--graph-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    MyGAT = ensure_import_official(args.official_ta_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows = encode_rows(load_jsonl(os.path.join(args.graph_data_dir, "train.jsonl")), args.embedding_model)
    test_rows = encode_rows(load_jsonl(os.path.join(args.graph_data_dir, "balanced_common.jsonl")), args.embedding_model)
    stats_path = os.path.join(args.graph_data_dir, "stats.json")
    agent_label_policy = "unknown"
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as handle:
            agent_label_policy = json.load(handle).get("agent_label_policy", "unknown")
    model = train_model(MyGAT, train_rows, device, args.epochs, args.lr, args.hidden_dim)
    metrics = evaluate(model, test_rows, device, os.path.join(args.output_dir, "predictions.jsonl"), agent_label_policy)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
