from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Keep the sibling import valid both for direct execution and for the dynamic
# import used by the package preflight self-test.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import v19_component_gnn_multitask as common


OFFICIAL_COMMITS = {
    "blindguard": "1889c20a326ba9ba9a6982744d473626e74f9986",
    "xgguard": "86e1121512f76800f80d4687e492c7f99f049929",
}
METHOD_NAMES = {"blindguard": "BlindGuard", "xgguard": "XG-Guard"}
CACHE_SCHEMA = "v19-unsupervised-bi-level-zero-truncation-v2"


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_commit(path: Path, kind: str) -> str:
    value = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if value != OFFICIAL_COMMITS[kind]:
        raise RuntimeError(f"Wrong official {kind} commit: {value} != {OFFICIAL_COMMITS[kind]}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def encode_bilevel(raw_graphs: list[dict], cache_path: Path, contract: dict) -> list[dict]:
    contract_path = cache_path.with_suffix(".contract.json")
    if cache_path.is_file() and contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) == contract:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        raise RuntimeError(f"Feature cache contract mismatch: {cache_path}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        common.ENCODER_MODEL, revision=common.ENCODER_REVISION,
        cache_folder=os.environ.get("HF_HOME"), device="cuda",
    )
    encoder.eval()
    tokenizer = encoder.tokenizer
    capacity = int(encoder.max_seq_length) - int(tokenizer.num_special_tokens_to_add(pair=False))
    pieces = list(dict.fromkeys(p for graph in raw_graphs for group in graph["pieces"] for p in group))
    old_limit = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    encoded_piece: dict[str, dict] = {}
    max_tokens = chunk_count = chunked = 0
    for piece in tqdm(pieces, desc="xgguard_zero_truncation_encode"):
        ids = tokenizer.encode(piece, add_special_tokens=False, truncation=False)
        max_tokens = max(max_tokens, len(ids))
        chunks = [ids[i:i + capacity] for i in range(0, len(ids), capacity)] or [[]]
        chunked += len(chunks) > 1
        sentence_vectors, token_vectors, weights = [], [], []
        for start in range(0, len(chunks), 64):
            current = chunks[start:start + 64]
            prepared = [{
                "input_ids": [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id],
                "attention_mask": [1] * (len(chunk) + 2),
                "token_type_ids": [0] * (len(chunk) + 2),
            } for chunk in current]
            features = tokenizer.pad(prepared, padding=True, return_tensors="pt")
            features = {key: value.to(encoder.device) for key, value in features.items()}
            with torch.no_grad():
                output = encoder(features)
            sent = F.normalize(output["sentence_embedding"], p=2, dim=1).cpu()
            toks = output["token_embeddings"].cpu()
            masks = features["attention_mask"].cpu()
            for local, chunk in enumerate(current):
                sentence_vectors.append(sent[local])
                # Exclude special and padding tokens; every source token remains represented.
                # Candidate anomaly scoring in the released XG-Guard code uses
                # the mean token score.  Because its scorer is linear in every
                # token embedding, retaining each chunk mean and its token
                # weight is an exact sufficient statistic for that score while
                # avoiding a many-GB token cache for the long V19 records.
                current_tokens = toks[local, 1:1 + len(chunk)]
                token_vectors.append(current_tokens.mean(0, keepdim=True) if len(chunk) else sentence_vectors[-1].unsqueeze(0))
                weights.append(max(len(chunk), 1))
                chunk_count += 1
        sentence = torch.stack(sentence_vectors)
        weight = torch.tensor(weights, dtype=sentence.dtype).unsqueeze(1)
        sentence = F.normalize((sentence * weight).sum(0) / weight.sum(), p=2, dim=0)
        tokens = torch.cat(token_vectors, dim=0)
        encoded_piece[piece] = {"sentence": sentence, "tokens": tokens,
                               "token_weights": torch.tensor(weights, dtype=torch.float32),
                               "span_texts": [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]}
    tokenizer.model_max_length = old_limit

    rows = []
    for graph in raw_graphs:
        sentences, candidate_tokens = [], []
        for group in graph["pieces"]:
            group_sent = F.normalize(torch.stack([encoded_piece[p]["sentence"] for p in group]).mean(0), p=2, dim=0)
            sentences.append(group_sent)
            vectors = torch.cat([encoded_piece[p]["tokens"] for p in group], dim=0)
            weights = torch.cat([encoded_piece[p]["token_weights"] for p in group], dim=0)
            span_texts = [text for p in group for text in encoded_piece[p]["span_texts"]]
            candidate_tokens.append({"vectors": vectors, "weights": weights, "span_texts": span_texts})
        item = {key: value for key, value in graph.items() if key != "pieces"}
        item["x_sentence"] = torch.stack(sentences)
        item["x_tokens"] = candidate_tokens
        rows.append(item)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache_path)
    json_dump(contract_path, contract)
    json_dump(cache_path.with_suffix(".encoding_audit.json"), {
        "status": "PASS", "zero_truncation": True, "unique_pieces": len(pieces),
        "encoded_chunks": chunk_count, "chunked_pieces": chunked,
        "max_piece_tokens": max_tokens, "token_capacity_per_chunk": capacity,
    })
    return rows


class BlindGuardModel(nn.Module):
    def __init__(self, official_dir: Path, dim: int, hidden: int, latent: int):
        super().__init__()
        sys.path.insert(0, str(official_dir.resolve()))
        from TAM import GATSCL  # type: ignore
        self.core = GATSCL(dim, hidden, latent, heads=8, dropout=0.0)

    def training_loss(self, row: dict, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        x = row["x_sentence"].to(device)
        edge = torch.as_tensor(row["edge_index"], dtype=torch.long, device=device)
        n = x.size(0)
        count = max(1, int(round(n * 3 / 8)))
        mask = torch.zeros(n, device=device)
        mask[torch.randperm(n, generator=generator, device=device)[:count]] = 1
        noise = F.normalize(torch.randn(x.shape, generator=generator, device=device), p=2, dim=1)
        corrupted = x + noise * (5.0 * torch.norm(x, dim=1, keepdim=True)) * mask[:, None]
        return self.core.neg_all(self.core.encode(corrupted, edge), mask)

    def scores(self, row: dict, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = row["x_sentence"].to(device)
        edge = torch.as_tensor(row["edge_index"], dtype=torch.long, device=device)
        z = self.core.encode(x, edge)
        adj = torch.eye(x.size(0), device=device)
        adj[edge[0], edge[1]] = 1.0
        # Official BlindGuard ranks the negative all-node similarity message.
        score = -self.core.inference_new(z, adj)
        return score, []


class GCNEncoder(nn.Module):
    """Exact default one-layer GCN used by official XG-Guard Ours.py."""
    def __init__(self, dim: int):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv = GCNConv(dim, dim)
        torch.nn.init.normal_(self.conv.lin.weight, mean=0.0, std=0.0005)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge)


class XGGuardModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.x_proj = GCNEncoder(dim)
        self.gnn = GCNEncoder(dim)

    def representation(self, row: dict, device: torch.device):
        sentence = row["x_sentence"].to(device)
        tokens = [value["vectors"].to(device) for value in row["x_tokens"]]
        weights = [value["weights"].to(device) for value in row["x_tokens"]]
        edge = torch.as_tensor(row["edge_index"], dtype=torch.long, device=device)
        sent_emb = self.x_proj(sentence, edge) + sentence
        token_mean = torch.stack([(value * weight[:, None]).sum(0) / weight.sum() for value, weight in zip(tokens, weights)])
        token_neighbour = self.gnn(sentence + token_mean, edge) + sentence
        token_emb = [value + token_neighbour[i] for i, value in enumerate(tokens)]
        token_node_means = torch.stack([(value * weight[:, None]).sum(0) / weight.sum() for value, weight in zip(token_emb, weights)])
        return sent_emb, token_emb, weights, sent_emb.mean(0), token_node_means.mean(0)

    @staticmethod
    def level_scores(sent_emb, token_emb, token_weights, sent_context, token_context):
        sentence = -(sent_emb @ sent_context)
        token_detail = [-(tokens @ token_context) for tokens in token_emb]
        token = torch.stack([(scores * weight).sum() / weight.sum() for scores, weight in zip(token_detail, token_weights)])
        return sentence, token, token_detail

    @staticmethod
    def fuse(sentence: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        def standardize(value):
            return (value - value.mean()) / value.std(unbiased=True).clamp_min(1e-6)
        sentence, token = standardize(sentence), standardize(token)
        return sentence + torch.mean(sentence * token) * token

    def training_loss(self, rows: list[dict], device: torch.device, alpha: float, generator: torch.Generator):
        reps = [self.representation(row, device) for row in rows]
        contexts_s = torch.stack([rep[3] for rep in reps])
        contexts_t = torch.stack([rep[4] for rep in reps])
        permutation = torch.randperm(len(rows), generator=generator, device=device)
        pos_sentence, pos_token, neg_sentence, neg_token = [], [], [], []
        for i, (sent, toks, weights, _, _) in enumerate(reps):
            ps, pt, _ = self.level_scores(sent, toks, weights, contexts_s[i], contexts_t[i])
            ns, nt, _ = self.level_scores(sent, toks, weights, contexts_s[permutation[i]], contexts_t[permutation[i]])
            pos_sentence.append(ps); pos_token.append(pt)
            neg_sentence.append(ns); neg_token.append(nt)
        # Official code standardizes the complete batch-by-node score tensor.
        # Concatenation is the same operation for variable-size V19 graphs.
        pos = self.fuse(torch.cat(pos_sentence), torch.cat(pos_token)) / 2
        neg = self.fuse(torch.cat(neg_sentence), torch.cat(neg_token)) / 2
        return F.binary_cross_entropy_with_logits(pos, torch.zeros_like(pos)) + alpha * F.binary_cross_entropy_with_logits(neg, torch.ones_like(neg))

    def scores(self, row: dict, device: torch.device):
        sent, toks, weights, context_s, context_t = self.representation(row, device)
        score_s, score_t, detail = self.level_scores(sent, toks, weights, context_s, context_t)
        return self.fuse(score_s, score_t), detail


def make_model(kind: str, official_dir: Path, dim: int, hidden: int, latent: int):
    if kind == "blindguard":
        return BlindGuardModel(official_dir, dim, hidden, latent)
    return XGGuardModel(dim)


def score_rows(model, rows: list[dict], device: torch.device) -> list[dict]:
    model.eval()
    output = []
    with torch.no_grad():
        for row in tqdm(rows, desc="score"):
            scores, detail = model.scores(row, device)
            values = scores.detach().cpu().float().numpy()
            explanations = []
            for cid, candidate, candidate_detail in zip(row["candidate_ids"], row["x_tokens"], detail):
                span_scores = candidate_detail.detach().cpu().float().numpy()
                top = np.argsort(-span_scores)[:3]
                explanations.append({"component_id": cid, "top_anomalous_spans": [
                    {"score": float(span_scores[i]), "text": candidate["span_texts"][int(i)]}
                    for i in top
                ]})
            output.append({"row": row, "scores": values, "graph_score": float(np.max(values)),
                           "explanations": explanations})
    return output


def verdict_from_score(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "clean_safe"
    return "attack_failed" if value < upper else "attack_success"


def calibrate_normal_only(scored: list[dict], lower_quantile: float, upper_quantile: float,
                          component_quantile: float) -> dict:
    graph_scores = np.asarray([item["graph_score"] for item in scored])
    candidate_scores = np.concatenate([item["scores"] for item in scored])
    lower = float(np.quantile(graph_scores, lower_quantile))
    upper = float(np.quantile(graph_scores, upper_quantile))
    if upper <= lower:
        upper = float(np.nextafter(lower, math.inf))
    return {"verdict_lower": lower, "verdict_upper": upper,
            "component_threshold": float(np.quantile(candidate_scores, component_quantile)),
            "source": "clean_safe training anomaly distribution only; no attack/validation labels",
            "lower_quantile": lower_quantile, "upper_quantile": upper_quantile,
            "component_quantile": component_quantile}


def build_records(scored, lower: float, upper: float, component_threshold: float) -> list[dict]:
    records = []
    for item in scored:
        row = item["row"]
        components = [cid for cid, score in zip(row["candidate_ids"], item["scores"]) if score >= component_threshold]
        records.append({
            "run_id": row["run_id"], "gold": row["gold_verdict"],
            "pred": verdict_from_score(item["graph_score"], lower, upper),
            "gold_scope": row["gold_scope"], "pred_scope": common.predicted_scope(components),
            "gold_components": row["gold_components"], "pred_components": components,
            "component_ids": row["candidate_ids"], "component_anomaly_scores": item["scores"].tolist(),
            "graph_anomaly_score": item["graph_score"], "fine_grained_explanations": item["explanations"],
        })
    return records


def metrics(records: list[dict]) -> dict:
    gold, pred = [r["gold"] for r in records], [r["pred"] for r in records]
    report = classification_report(gold, pred, labels=list(common.VERDICTS), zero_division=0, output_dict=True)
    return {
        "n": len(records), "three_class_accuracy": accuracy_score(gold, pred),
        "three_class_report": report,
        "localization": common.localization_summary(records, lambda row: row["gold"] == "attack_success"),
        "all_attacked_localization": common.localization_summary(records, lambda row: row["gold"] != "clean_safe"),
    }


def save_predictions(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_splits(data_dir: Path, cache_dir: Path, splits: tuple[str, ...], normal_only_train: bool = False):
    encoded, hashes = {}, {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        if actual != common.MARBLE_SHA256[split]:
            raise RuntimeError(f"Frozen MARBLE {split} hash mismatch: {actual}")
        hashes[split] = actual
        raw = common.build_raw_graphs(path)
        if split == "train" and normal_only_train:
            raw = [row for row in raw if row["gold_verdict"] == "clean_safe"]
        contract = {"data_sha256": actual, "encoder_model": common.ENCODER_MODEL,
                    "encoder_revision": common.ENCODER_REVISION, "schema": CACHE_SCHEMA}
        contract["row_filter"] = "clean_safe only" if split == "train" and normal_only_train else "all"
        encoded[split] = encode_bilevel(raw, cache_dir / f"{split}.pt", contract)
    return encoded, hashes


def train(args) -> None:
    if not (0.0 < args.component_quantile < 1.0 and
            0.0 < args.lower_quantile < args.upper_quantile < 1.0):
        raise ValueError("Require 0 < component quantile < 1 and 0 < lower < upper < 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device, output = torch.device("cuda"), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = official_commit(Path(args.official_dir), args.model_kind)
    encoded, hashes = load_splits(Path(args.data_dir), Path(args.cache_dir), ("train", "validation"), normal_only_train=True)
    normal = [row for row in encoded["train"] if row["gold_verdict"] == "clean_safe"]
    if not normal:
        raise RuntimeError("No clean_safe rows available for unsupervised normal-only training")
    seed_everything(args.seed)
    dim = int(normal[0]["x_sentence"].shape[1])
    model = make_model(args.model_kind, Path(args.official_dir), dim, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    resume_contract = {"model_kind": args.model_kind, "official_commit": commit,
                       "train_sha256": hashes["train"], "seed": args.seed, "epochs": args.epochs,
                       "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                       "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim, "alpha": args.alpha}
    history, best_loss, best_epoch, start_epoch = [], math.inf, 0, 1
    last_checkpoint = output / "last_checkpoint.pt"
    if last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if state.get("resume_contract") != resume_contract:
            raise RuntimeError("Existing checkpoint contract does not match this run")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"]); best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"]); start_epoch = int(state["epoch"]) + 1
        generator.set_state(state["cuda_generator_state"].cpu())
        print(json.dumps({"resume": "PASS", "start_epoch": start_epoch}))
    for epoch in range(start_epoch, args.epochs + 1):
        order = torch.randperm(len(normal), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        model.train(); total = 0.0; batches = 0
        for start in tqdm(range(0, len(order), args.batch_size), desc=f"{args.model_kind}_epoch_{epoch}"):
            batch = [normal[i] for i in order[start:start + args.batch_size]]
            if args.model_kind == "xgguard" and len(batch) == 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            if args.model_kind == "blindguard":
                loss = torch.stack([model.training_loss(row, device, generator) for row in batch]).mean()
            else:
                loss = model.training_loss(batch, device, args.alpha, generator)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {args.model_kind} loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu()); batches += 1
        epoch_loss = total / max(batches, 1)
        history.append({"epoch": epoch, "normal_only_train_loss": epoch_loss})
        print(json.dumps(history[-1]))
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            torch.save(model.state_dict(), output / "best_model.pt")
        torch.save({"resume_contract": resume_contract,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                    "cuda_generator_state": generator.get_state()}, output / "last_checkpoint.pt")
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device, weights_only=True))
    calibration_scores = score_rows(model, normal, device)
    calibration = calibrate_normal_only(calibration_scores, args.lower_quantile, args.upper_quantile,
                                        args.component_quantile)
    scored = score_rows(model, encoded["validation"], device)
    records = build_records(scored, calibration["verdict_lower"], calibration["verdict_upper"], calibration["component_threshold"])
    result = metrics(records)
    result.update({
        "mode": "v19_normal_only_graph_anomaly", "method": METHOD_NAMES[args.model_kind],
        "model_kind": args.model_kind, "official_commit": commit,
        "encoder_model": common.ENCODER_MODEL, "encoder_revision": common.ENCODER_REVISION,
        "data_file": str((Path(args.data_dir) / "validation.jsonl").resolve()),
        "data_sha256": hashes["validation"], "dataset_role": "validation",
        "training_rows": len(normal), "training_filter": "train verdict == clean_safe; labels excluded from optimization",
        "calibration": calibration, "selected_epoch": best_epoch,
        "adaptation_disclosure": "Native anomaly scores; normal-training-distribution two-threshold projection to V19 3-way labels and threshold projection to G/N/E/T candidates.",
        "input_policy": "V19 user message only; zero-truncation candidate graph projection",
    })
    json_dump(output / "metrics.json", result); save_predictions(output / "predictions.jsonl", records)
    checkpoint_hash = sha256_file(output / "best_model.pt")
    json_dump(output / "TRAIN_CONTRACT.json", {
        "model_kind": args.model_kind, "official_commit": commit, "encoder_model": common.ENCODER_MODEL,
        "encoder_revision": common.ENCODER_REVISION, "train_sha256": hashes["train"],
        "validation_sha256": hashes["validation"], "test_accessed": False, "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
        "alpha": args.alpha, "normal_only_training_rows": len(normal), "best_epoch": best_epoch,
        "lower_quantile": args.lower_quantile, "upper_quantile": args.upper_quantile,
        "component_quantile": args.component_quantile,
        "calibration": calibration, "history": history, "best_model_sha256": checkpoint_hash,
    })
    print(json.dumps(result, indent=2))


def final_test(args) -> None:
    if args.sealed_test_ack != "FINAL_ONCE":
        raise RuntimeError("Final test requires --sealed-test-ack FINAL_ONCE")
    checkpoint_dir, output = Path(args.checkpoint_dir), Path(args.output_dir)
    if output.exists():
        raise RuntimeError("Final-test output directory must not already exist")
    output.mkdir(parents=True)
    contract = json.loads((checkpoint_dir / "TRAIN_CONTRACT.json").read_text(encoding="utf-8"))
    kind = contract["model_kind"]
    official_commit(Path(args.official_dir), kind)
    checkpoint = checkpoint_dir / "best_model.pt"
    if sha256_file(checkpoint) != contract["best_model_sha256"]:
        raise RuntimeError("Best-model hash mismatch")
    rows, hashes = load_splits(Path(args.data_dir), Path(args.cache_dir), ("test",))
    device = torch.device("cuda")
    dim = int(rows["test"][0]["x_sentence"].shape[1])
    model = make_model(kind, Path(args.official_dir), dim, contract["hidden_dim"], contract["latent_dim"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    scored = score_rows(model, rows["test"], device); cal = contract["calibration"]
    records = build_records(scored, cal["verdict_lower"], cal["verdict_upper"], cal["component_threshold"])
    result = metrics(records)
    result.update({"mode": "v19_normal_only_graph_anomaly", "method": METHOD_NAMES[kind],
                   "model_kind": kind, "official_commit": contract["official_commit"],
                   "data_sha256": hashes["test"], "dataset_role": "test", "calibration": cal,
                   "checkpoint_sha256": contract["best_model_sha256"]})
    json_dump(output / "SEALED_TEST_CONSUMED.json", {"test_sha256": hashes["test"], "rows": len(records)})
    json_dump(output / "metrics.json", result); save_predictions(output / "predictions.jsonl", records)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train-validation")
    train_p.add_argument("--model-kind", choices=["blindguard", "xgguard"], required=True)
    train_p.add_argument("--official-dir", required=True); train_p.add_argument("--data-dir", required=True)
    train_p.add_argument("--cache-dir", required=True); train_p.add_argument("--output-dir", required=True)
    train_p.add_argument("--epochs", type=int, default=50); train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=1e-5); train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--hidden-dim", type=int, default=512); train_p.add_argument("--latent-dim", type=int, default=256)
    train_p.add_argument("--alpha", type=float, default=1e-4); train_p.add_argument("--seed", type=int, default=3701)
    train_p.add_argument("--lower-quantile", type=float, default=0.95)
    train_p.add_argument("--upper-quantile", type=float, default=0.99)
    train_p.add_argument("--component-quantile", type=float, default=0.90)
    test_p = sub.add_parser("final-test")
    test_p.add_argument("--checkpoint-dir", required=True); test_p.add_argument("--official-dir", required=True)
    test_p.add_argument("--data-dir", required=True); test_p.add_argument("--cache-dir", required=True)
    test_p.add_argument("--output-dir", required=True); test_p.add_argument("--sealed-test-ack", choices=["FINAL_ONCE"], required=True)
    args = parser.parse_args()
    train(args) if args.command == "train-validation" else final_test(args)


if __name__ == "__main__":
    main()
