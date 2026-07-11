import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def batch_encode(model, texts, batch_size=256):
    embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode_text"):
        chunk = texts[start : start + batch_size]
        embs.extend(model.encode(chunk, normalize_embeddings=True, show_progress_bar=False).tolist())
    return embs


def embed_dataset(data_dir, output_dir, model_name):
    from sentence_transformers import SentenceTransformer

    os.makedirs(output_dir, exist_ok=True)
    encoder = SentenceTransformer(model_name)
    for split in ["train", "test", "all"]:
        src = os.path.join(data_dir, f"{split}.jsonl")
        dst = os.path.join(output_dir, f"{split}.jsonl")
        if os.path.exists(dst):
            continue
        rows = load_jsonl(src)
        node_texts = []
        edge_texts = []
        for row in rows:
            node_texts.extend(row["candidate_texts"])
            edge_texts.extend(row["edge_texts"])
        node_embs = batch_encode(encoder, node_texts)
        edge_embs = batch_encode(encoder, edge_texts)
        ni = 0
        ei = 0
        with open(dst, "w", encoding="utf-8") as handle:
            for row in rows:
                n = len(row["candidate_texts"])
                e = len(row["edge_texts"])
                out = dict(row)
                out["node_embeddings"] = node_embs[ni : ni + n]
                out["edge_embeddings"] = edge_embs[ei : ei + e]
                del out["candidate_texts"]
                del out["edge_texts"]
                ni += n
                ei += e
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")


class EdgeAwareGATLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.15):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = dropout

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        q = self.q(h[dst])
        k = self.k(h[src]) + self.edge(edge_attr)
        v = self.v(h[src]) + self.edge(edge_attr)
        score = (q * k).sum(-1) / (h.size(-1) ** 0.5)
        # Small-graph segmented softmax without torch_scatter.
        attn = torch.zeros_like(score)
        for node in torch.unique(dst):
            mask = dst == node
            attn[mask] = torch.softmax(score[mask], dim=0)
        msg = v * F.dropout(attn.unsqueeze(-1), p=self.dropout, training=self.training)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return F.relu(self.out(torch.cat([h, agg], dim=-1)))


class StrongGraphTeacher(nn.Module):
    def __init__(self, text_dim, type_dim, hidden_dim=192, layers=2, dropout=0.15):
        super().__init__()
        self.input = nn.Linear(text_dim + type_dim, hidden_dim)
        self.layers = nn.ModuleList(EdgeAwareGATLayer(hidden_dim, text_dim, dropout=dropout) for _ in range(layers))
        self.component_head = nn.Linear(hidden_dim, 1)
        self.run_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb, type_feat, edge_index, edge_emb):
        h = F.relu(self.input(torch.cat([node_emb, type_feat], dim=-1)))
        for layer in self.layers:
            h = layer(h, edge_index, edge_emb)
        component_logits = self.component_head(h).squeeze(-1)
        run_logit = self.run_head(h.max(dim=0).values).squeeze(-1)
        return component_logits, run_logit


def to_tensors(row, device):
    node_emb = torch.tensor(row["node_embeddings"], dtype=torch.float32, device=device)
    type_feat = torch.tensor(row["candidate_type_features"], dtype=torch.float32, device=device)
    edge_emb = torch.tensor(row["edge_embeddings"], dtype=torch.float32, device=device)
    if row["edge_index"]:
        edge_index = torch.tensor(row["edge_index"], dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    y_comp = torch.tensor(row["component_labels"], dtype=torch.float32, device=device)
    y_run = torch.tensor(float(row["run_label"]), dtype=torch.float32, device=device)
    return node_emb, type_feat, edge_index, edge_emb, y_comp, y_run


def normal_centroid(rows, device):
    vectors = []
    for row in rows:
        if row.get("run_label") != 0:
            continue
        node_emb = torch.tensor(row["node_embeddings"], dtype=torch.float32, device=device)
        labels = row.get("component_labels", [])
        for idx, label in enumerate(labels):
            if label == 0:
                vectors.append(node_emb[idx])
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def anomaly_scores(node_emb, centroid):
    if centroid is None:
        return torch.zeros(node_emb.size(0), device=node_emb.device)
    dist = torch.norm(node_emb - centroid.unsqueeze(0), dim=-1)
    if dist.numel() <= 1:
        return torch.zeros_like(dist)
    return (dist - dist.min()) / (dist.max() - dist.min()).clamp_min(1e-6)


def evaluate(model, rows, device, output_path=None, centroid=None):
    model.eval()
    gold_run, pred_run = [], []
    gold_comp, pred_comp = [], []
    records = []
    with torch.no_grad():
        for row in rows:
            node_emb, type_feat, edge_index, edge_emb, y_comp, y_run = to_tensors(row, device)
            comp_logits, run_logit = model(node_emb, type_feat, edge_index, edge_emb)
            supervised_scores_t = torch.sigmoid(comp_logits)
            anomaly_scores_t = anomaly_scores(node_emb, centroid)
            comp_scores_t = 0.75 * supervised_scores_t + 0.25 * anomaly_scores_t
            comp_scores = comp_scores_t.detach().cpu().tolist()
            run_score = float(torch.sigmoid(run_logit).detach().cpu())
            comp_pred = [1 if score >= 0.5 else 0 for score in comp_scores]
            run_pred = 1 if run_score >= 0.5 else 0
            gold_run.append(int(row["run_label"]))
            pred_run.append(run_pred)
            gold_comp.extend(row["component_labels"])
            pred_comp.extend(comp_pred)
            ranking = sorted(
                [{"id": cid, "score": round(score, 6), "type": ctype} for cid, ctype, score in zip(row["candidate_ids"], row["candidate_types"], comp_scores)],
                key=lambda item: item["score"],
                reverse=True,
            )
            records.append({
                "run_id": row["run_id"],
                "run_score": round(run_score, 6),
                "run_pred": run_pred,
                "run_gold": row["run_label"],
                "component_ranking": ranking,
            })
    metrics = {
        "run_accuracy": accuracy_score(gold_run, pred_run),
        "run_report": classification_report(gold_run, pred_run, labels=[0, 1], target_names=["safe", "unsafe"], zero_division=0, output_dict=True),
        "component_report": classification_report(gold_comp, pred_comp, labels=[0, 1], target_names=["not_target", "target"], zero_division=0, output_dict=True),
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-data-dir", required=True)
    parser.add_argument("--embedded-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    embed_dataset(args.gnn_data_dir, args.embedded_data_dir, args.embedding_model)
    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = load_jsonl(os.path.join(args.embedded_data_dir, "train.jsonl"))
    test_rows = load_jsonl(os.path.join(args.embedded_data_dir, "test.jsonl"))
    text_dim = len(train_rows[0]["node_embeddings"][0])
    type_dim = len(train_rows[0]["candidate_type_features"][0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StrongGraphTeacher(text_dim, type_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    best_path = os.path.join(args.output_dir, "strong_edge_gat.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_rows)
        total = 0.0
        for row in tqdm(train_rows, desc=f"strong_edge_gat_epoch_{epoch}"):
            node_emb, type_feat, edge_index, edge_emb, y_comp, y_run = to_tensors(row, device)
            comp_logits, run_logit = model(node_emb, type_feat, edge_index, edge_emb)
            comp_loss = F.binary_cross_entropy_with_logits(comp_logits, y_comp, pos_weight=torch.tensor(4.0, device=device))
            run_loss = F.binary_cross_entropy_with_logits(run_logit, y_run)
            loss = comp_loss + run_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        centroid = normal_centroid(train_rows, device)
        metrics = evaluate(model, test_rows, device, centroid=centroid)
        score = metrics["run_report"]["unsafe"]["f1-score"] + metrics["component_report"]["target"]["f1-score"]
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
        print(json.dumps({"epoch": epoch, "loss": total / len(train_rows), "metrics": metrics}, indent=2))

    model.load_state_dict(torch.load(best_path, map_location=device))
    centroid = normal_centroid(train_rows, device)
    metrics = evaluate(model, test_rows, device, os.path.join(args.output_dir, "test_scores.jsonl"), centroid=centroid)
    all_rows = load_jsonl(os.path.join(args.embedded_data_dir, "all.jsonl"))
    evaluate(model, all_rows, device, os.path.join(args.output_dir, "all_scores.jsonl"), centroid=centroid)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
