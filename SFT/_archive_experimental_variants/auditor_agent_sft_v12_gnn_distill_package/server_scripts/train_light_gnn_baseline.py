import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


class LightGraphDetector(nn.Module):
    def __init__(self, in_dim, hidden_dim=96, layers=2):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers))
        self.component_head = nn.Linear(hidden_dim, 1)
        self.run_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.input(x))
        if edge_index.numel() == 0:
            edge_index = torch.arange(x.size(0), device=x.device).repeat(2, 1)
        src, dst = edge_index[0], edge_index[1]
        for layer in self.layers:
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])
            deg = torch.zeros(h.size(0), device=h.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp_min(1).unsqueeze(-1)
            h = F.relu(layer(torch.cat([h, agg], dim=-1)))
        component_logits = self.component_head(h).squeeze(-1)
        run_logit = self.run_head(h.max(dim=0).values).squeeze(-1)
        return component_logits, run_logit


def to_tensors(row, device):
    x = torch.tensor(row["features"], dtype=torch.float32, device=device)
    if row["edge_index"]:
        edge_index = torch.tensor(row["edge_index"], dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    y_comp = torch.tensor(row["component_labels"], dtype=torch.float32, device=device)
    y_run = torch.tensor(float(row["run_label"]), dtype=torch.float32, device=device)
    return x, edge_index, y_comp, y_run


def evaluate(model, rows, device, output_path=None):
    model.eval()
    gold_run, pred_run = [], []
    gold_comp, pred_comp = [], []
    records = []
    with torch.no_grad():
        for row in rows:
            x, edge_index, y_comp, y_run = to_tensors(row, device)
            comp_logits, run_logit = model(x, edge_index)
            comp_scores = torch.sigmoid(comp_logits).detach().cpu().tolist()
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train_rows = load_jsonl(os.path.join(args.gnn_data_dir, "train.jsonl"))
    test_rows = load_jsonl(os.path.join(args.gnn_data_dir, "test.jsonl"))
    in_dim = len(train_rows[0]["features"][0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightGraphDetector(in_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    best_path = os.path.join(args.output_dir, "light_gnn.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_rows)
        total = 0.0
        for row in tqdm(train_rows, desc=f"light_gnn_epoch_{epoch}"):
            x, edge_index, y_comp, y_run = to_tensors(row, device)
            comp_logits, run_logit = model(x, edge_index)
            comp_loss = F.binary_cross_entropy_with_logits(comp_logits, y_comp, pos_weight=torch.tensor(4.0, device=device))
            run_loss = F.binary_cross_entropy_with_logits(run_logit, y_run)
            loss = comp_loss + run_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        metrics = evaluate(model, test_rows, device)
        score = metrics["run_report"]["unsafe"]["f1-score"] + metrics["component_report"]["target"]["f1-score"]
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
        print(json.dumps({"epoch": epoch, "loss": total / len(train_rows), "metrics": metrics}, indent=2))

    model.load_state_dict(torch.load(best_path, map_location=device))
    metrics = evaluate(model, test_rows, device, os.path.join(args.output_dir, "test_scores.jsonl"))
    evaluate(model, load_jsonl(os.path.join(args.gnn_data_dir, "all.jsonl")), device, os.path.join(args.output_dir, "all_scores.jsonl"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
