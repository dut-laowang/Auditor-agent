from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


VERDICTS = ("clean_safe", "attack_failed", "attack_success")
VERDICT_TO_ID = {name: idx for idx, name in enumerate(VERDICTS)}
TYPE_ORDER = ("global", "node", "edge", "tool", "unknown")
MARBLE_SHA256 = {
    "train": "d49ec56577a80fab3be360ae9cb1b90e2d751ce686ffa0f0e2064b2d05d0a932",
    "validation": "2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e",
    "test": "bee77d962f66f5481e88d89b49b83b3ea9a449e48d776b669ebadd731417167f",
}
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FORBIDDEN_USER_TEXT = re.compile(
    r"success_markers?|attack_metadata|source_final_label|semantic_consensus|"
    r"judge_conclusion|gold_label|\battack_id\b|\bpayload_id\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def parse_row(row: dict) -> tuple[dict, dict, str, list[str]]:
    messages = row.get("messages") or []
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("V19 row must contain system/user/assistant in that order")
    user_text = messages[1].get("content")
    if not isinstance(user_text, str) or FORBIDDEN_USER_TEXT.search(user_text):
        raise ValueError("Privileged or malformed content found in V19 GNN-visible user input")
    user = json.loads(user_text)
    assistant = json.loads(messages[2]["content"])
    verdict = assistant["decision"]["verdict"]
    components = assistant.get("localization", {}).get("component_ids", []) or []
    if verdict not in VERDICT_TO_ID:
        raise ValueError(f"Unknown verdict: {verdict}")
    return user, assistant, verdict, [str(value) for value in components]


def candidate_type(candidate: dict) -> str:
    kind = str(candidate.get("type") or "unknown")
    return kind if kind in TYPE_ORDER else "unknown"


def event_index(user: dict) -> dict[str, dict]:
    evidence = user.get("run_evidence", {})
    return {
        str(event["id"]): event
        for event in evidence.get("observed", []) + evidence.get("final_output", [])
        if isinstance(event, dict) and event.get("id")
    }


def candidate_pieces(user: dict, candidate: dict, events_by_id: dict[str, dict]) -> list[str]:
    kind = candidate_type(candidate)
    structural = [f"candidate type: {kind}"]
    for key in ("id", "description", "agent", "tool", "source", "target"):
        value = candidate.get(key)
        if value is not None:
            structural.append(f"{key}: {value}")
    pieces = ["; ".join(structural)]
    refs = []
    for key in ("event_refs", "local_event_refs", "incoming_event_refs", "outgoing_event_refs"):
        refs.extend(str(value) for value in candidate.get(key, []) or [])
    unique_refs = list(dict.fromkeys(refs))
    for ref in unique_refs:
        event = events_by_id.get(ref)
        if not event:
            continue
        header = " ".join(
            f"{key}={event[key]}"
            for key in ("type", "agent", "tool", "source_agent", "target_agent")
            if event.get(key) is not None
        )
        text = str(event.get("text") or "")
        pieces.append((header + " " + text).strip())
    if kind == "global":
        task = user.get("task", {})
        pieces.append(f"scenario={task.get('scenario')} task={task.get('goal', '')}")
        # The global candidate is the graph-level evidence collector.  Include
        # every otherwise-unreferenced observable event so the GNN receives the
        # complete V19 user input rather than an accidental candidate-only
        # subset. Candidate-specific nodes still retain their exact projections.
        for ref, event in events_by_id.items():
            if ref in unique_refs:
                continue
            header = " ".join(
                f"{key}={event[key]}"
                for key in ("type", "agent", "tool", "source_agent", "target_agent")
                if event.get(key) is not None
            )
            pieces.append((header + " " + str(event.get("text") or "")).strip())
    return [piece for piece in pieces if piece.strip()]


def component_edges(user: dict, candidates: list[dict]) -> np.ndarray:
    ids = [str(candidate["id"]) for candidate in candidates]
    by_id = {cid: idx for idx, cid in enumerate(ids)}
    edges: set[tuple[int, int]] = {(idx, idx) for idx in range(len(ids))}

    def link(left: str, right: str) -> None:
        if left in by_id and right in by_id:
            edges.add((by_id[left], by_id[right]))
            edges.add((by_id[right], by_id[left]))

    if "G::run" in by_id:
        for cid in ids:
            link("G::run", cid)
    for edge in user.get("graph", {}).get("edges", []) or []:
        link(f"N::{edge.get('source')}", f"N::{edge.get('target')}")
    for candidate in candidates:
        cid = str(candidate["id"])
        kind = candidate_type(candidate)
        if kind == "edge":
            link(cid, f"N::{candidate.get('source')}")
            link(cid, f"N::{candidate.get('target')}")
        elif kind == "tool":
            link(cid, f"N::{candidate.get('agent')}")
    ordered = sorted(edges)
    return np.asarray(ordered, dtype=np.int64).T


def build_raw_graphs(path: Path) -> list[dict]:
    graphs = []
    for row in tqdm(load_jsonl(path), desc=f"parse_{path.stem}"):
        user, assistant, verdict, gold_components = parse_row(row)
        candidates = user.get("graph_candidates", []) or []
        candidate_ids = [str(candidate["id"]) for candidate in candidates]
        if not candidates or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Missing or duplicate graph_candidates")
        missing = set(gold_components) - set(candidate_ids)
        if missing:
            raise ValueError(f"Gold components absent from candidates: {sorted(missing)}")
        events_by_id = event_index(user)
        graphs.append(
            {
                "run_id": row.get("metadata", {}).get("run_id"),
                "candidate_ids": candidate_ids,
                "candidate_types": [candidate_type(candidate) for candidate in candidates],
                "pieces": [candidate_pieces(user, candidate, events_by_id) for candidate in candidates],
                "edge_index": component_edges(user, candidates),
                "gold_verdict": verdict,
                "gold_scope": str(assistant.get("localization", {}).get("scope") or "none"),
                "gold_components": gold_components,
            }
        )
    return graphs


def encode_graphs(raw_graphs: list[dict], cache_path: Path, cache_contract: dict) -> list[dict]:
    contract_path = cache_path.with_suffix(".contract.json")
    if cache_path.is_file() and contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) == cache_contract:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        raise RuntimeError(f"Feature cache contract mismatch: {cache_path}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        ENCODER_MODEL,
        revision=ENCODER_REVISION,
        cache_folder=os.environ.get("HF_HOME"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    unique_pieces = list(
        dict.fromkeys(piece for graph in raw_graphs for pieces in graph["pieces"] for piece in pieces)
    )
    embeddings = encoder.encode(
        unique_pieces,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    lookup = {piece: embeddings[idx] for idx, piece in enumerate(unique_pieces)}
    encoded = []
    for graph in raw_graphs:
        rows = []
        for kind, pieces in zip(graph["candidate_types"], graph["pieces"]):
            text_feature = np.mean([lookup[piece] for piece in pieces], axis=0)
            one_hot = np.zeros(len(TYPE_ORDER), dtype=np.float32)
            one_hot[TYPE_ORDER.index(kind)] = 1.0
            rows.append(np.concatenate([text_feature.astype(np.float32), one_hot]))
        item = {key: value for key, value in graph.items() if key != "pieces"}
        item["x"] = np.asarray(rows, dtype=np.float32)
        encoded.append(item)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoded, cache_path)
    json_dump(contract_path, cache_contract)
    return encoded


def graph_tensors(row: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.tensor(row["x"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(row["edge_index"], dtype=torch.long, device=device)
    verdict = torch.tensor(VERDICT_TO_ID[row["gold_verdict"]], dtype=torch.long, device=device)
    gold = set(row["gold_components"])
    components = torch.tensor(
        [1.0 if cid in gold else 0.0 for cid in row["candidate_ids"]],
        dtype=torch.float32,
        device=device,
    )
    return x, edge_index, verdict, components


class OfficialGATEncoder(nn.Module):
    def __init__(self, official_dir: Path, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        sys.path.insert(0, str(official_dir.resolve()))
        from model import MyGAT  # type: ignore

        self.model = MyGAT(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=latent_dim,
            heads=8,
            num_layers=2,
            edge_dim=(1, in_dim),
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_attr = x[edge_index[1]].unsqueeze(1)
        return F.relu(self.model(x, edge_index, edge_attr))


class OfficialTAMEncoder(nn.Module):
    def __init__(self, official_dir: Path, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        sys.path.insert(0, str(official_dir.resolve()))
        from TAM import TAMModel  # type: ignore

        self.model = TAMModel(in_dim, hidden_dim, latent_dim, dropout=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.model(x, edge_index)
        return F.relu(embeddings)


class V19MultiTaskGNN(nn.Module):
    def __init__(self, kind: str, official_dir: Path, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        if kind == "gat":
            self.encoder = OfficialGATEncoder(official_dir, in_dim, hidden_dim, latent_dim)
        elif kind == "tam":
            self.encoder = OfficialTAMEncoder(official_dir, in_dim, hidden_dim, latent_dim)
        else:
            raise ValueError(kind)
        self.loc_head = nn.Linear(latent_dim, 1)
        self.verdict_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 3)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        node_embeddings = self.encoder(x, edge_index)
        pooled = torch.cat([node_embeddings.mean(dim=0), node_embeddings.max(dim=0).values])
        return self.verdict_head(pooled), self.loc_head(node_embeddings).squeeze(-1)


def predicted_scope(candidate_ids: list[str]) -> str:
    if not candidate_ids:
        return "none"
    kinds = []
    for cid in candidate_ids:
        prefix = cid.split("::", 1)[0]
        kinds.append({"G": "global", "N": "node", "E": "edge", "T": "tool"}.get(prefix, "unknown"))
    return kinds[0] if len(set(kinds)) == 1 else "multi"


def safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def localization_summary(records: list[dict], selector) -> dict:
    rows = [row for row in records if selector(row) and row["gold_components"]]
    tp = fp = fn = hit = exact = scope = 0
    for row in rows:
        gold = set(row["gold_components"])
        pred = set(row["pred_components"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        hit += bool(gold & pred)
        exact += gold == pred
        scope += row["gold_scope"] == row["pred_scope"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "n": len(rows),
        "component_micro_precision": precision,
        "component_micro_recall": recall,
        "component_micro_f1": f1,
        "component_hit_rate": safe_div(hit, len(rows)),
        "component_exact_match": safe_div(exact, len(rows)),
        "scope_accuracy": safe_div(scope, len(rows)),
    }


def evaluate(model, rows: list[dict], device: torch.device, threshold: float) -> tuple[dict, list[dict]]:
    records = []
    model.eval()
    with torch.no_grad():
        for row in tqdm(rows, desc="evaluate"):
            x, edge_index, _, _ = graph_tensors(row, device)
            verdict_logits, component_logits = model(x, edge_index)
            pred_verdict = VERDICTS[int(verdict_logits.argmax().item())]
            probabilities = torch.sigmoid(component_logits).cpu().tolist()
            pred_components = [
                cid for cid, probability in zip(row["candidate_ids"], probabilities) if probability >= threshold
            ]
            records.append(
                {
                    "run_id": row["run_id"],
                    "gold": row["gold_verdict"],
                    "pred": pred_verdict,
                    "gold_scope": row["gold_scope"],
                    "pred_scope": predicted_scope(pred_components),
                    "gold_components": row["gold_components"],
                    "pred_components": pred_components,
                    "component_ids": row["candidate_ids"],
                    "component_probabilities": probabilities,
                }
            )
    gold = [row["gold"] for row in records]
    pred = [row["pred"] for row in records]
    primary_loc = localization_summary(records, lambda row: row["gold"] == "attack_success")
    all_attacked = localization_summary(records, lambda row: row["gold"] != "clean_safe")
    gated_records = []
    for row in records:
        copy = dict(row)
        if row["pred"] != "attack_success":
            copy["pred_components"] = []
            copy["pred_scope"] = "none"
        gated_records.append(copy)
    end_to_end = localization_summary(gated_records, lambda row: row["gold"] == "attack_success")
    report = classification_report(gold, pred, labels=list(VERDICTS), zero_division=0, output_dict=True)
    binary_gold = ["unsafe" if value == "attack_success" else "safe" for value in gold]
    binary_pred = ["unsafe" if value == "attack_success" else "safe" for value in pred]
    metrics = {
        "n": len(records),
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "confusion_matrix": {
            left: {right: sum(g == left and p == right for g, p in zip(gold, pred)) for right in VERDICTS}
            for left in VERDICTS
        },
        "three_class_accuracy": accuracy_score(gold, pred),
        "three_class_report": report,
        "binary_accuracy": accuracy_score(binary_gold, binary_pred),
        "binary_report": classification_report(
            binary_gold, binary_pred, labels=["safe", "unsafe"], zero_division=0, output_dict=True
        ),
        "localization": {
            "n_attack_success_with_gold_components": primary_loc.pop("n"),
            **primary_loc,
            "localization_policy": "source attack-placement candidate projection",
        },
        "additional_localization": {
            "all_attacked": all_attacked,
            "verdict_gated_attack_success": end_to_end,
        },
    }
    return metrics, records


def select_threshold(model, rows, device) -> float:
    best = (-1.0, 0.5)
    for threshold in [value / 100 for value in range(10, 91, 5)]:
        metrics, _ = evaluate(model, rows, device, threshold)
        score = metrics["localization"]["component_micro_f1"]
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def save_predictions(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_model(args, in_dim: int, device: torch.device):
    return V19MultiTaskGNN(
        args.model_kind, Path(args.official_dir), in_dim, args.hidden_dim, args.latent_dim
    ).to(device)


def train(args) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    encoded = {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        if actual != MARBLE_SHA256[split]:
            raise RuntimeError(f"Frozen MARBLE {split} hash mismatch: {actual}")
        hashes[split] = actual
        raw = build_raw_graphs(path)
        contract = {
            "data_sha256": actual,
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": ENCODER_REVISION,
            "candidate_graph_schema": "v19-component-graph-v1",
        }
        encoded[split] = encode_graphs(raw, cache_dir / f"{split}.pt", contract)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    in_dim = int(encoded["train"][0]["x"].shape[1])
    model = make_model(args, in_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    verdict_counts = Counter(row["gold_verdict"] for row in encoded["train"])
    class_weights = torch.tensor(
        [len(encoded["train"]) / (3 * verdict_counts[name]) for name in VERDICTS],
        dtype=torch.float32,
        device=device,
    )
    positives = sum(len(row["gold_components"]) for row in encoded["train"] if row["gold_verdict"] != "clean_safe")
    candidates = sum(len(row["candidate_ids"]) for row in encoded["train"])
    pos_weight = torch.tensor(min(max((candidates - positives) / max(positives, 1), 1.0), 30.0), device=device)
    best_score = -math.inf
    best_epoch = 0
    start_epoch = 1
    last_path = output_dir / "last_checkpoint.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_score = float(state["best_score"])
    history = list(state.get("history", [])) if last_path.is_file() else []
    for epoch in range(start_epoch, args.epochs + 1):
        order = list(range(len(encoded["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(tqdm(order, desc=f"{args.model_kind}_epoch_{epoch}"), 1):
            x, edge_index, verdict, components = graph_tensors(encoded["train"][index], device)
            verdict_logits, component_logits = model(x, edge_index)
            verdict_loss = F.cross_entropy(verdict_logits.unsqueeze(0), verdict.unsqueeze(0), weight=class_weights)
            component_loss = F.binary_cross_entropy_with_logits(component_logits, components, pos_weight=pos_weight)
            loss = (verdict_loss + args.localization_loss_weight * component_loss) / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * args.grad_accum
        validation_metrics, _ = evaluate(model, encoded["validation"], device, 0.5)
        macro_f1 = validation_metrics["three_class_report"]["macro avg"]["f1-score"]
        loc_f1 = validation_metrics["localization"]["component_micro_f1"]
        selection_score = (macro_f1 + loc_f1) / 2
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(order),
            "validation_macro_f1": macro_f1,
            "validation_localization_f1_at_0.5": loc_f1,
            "selection_score": selection_score,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))
        if selection_score > best_score:
            best_score, best_epoch = selection_score, epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": history,
            },
            last_path,
        )
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    threshold = select_threshold(model, encoded["validation"], device)
    metrics, records = evaluate(model, encoded["validation"], device, threshold)
    official_commit = os.popen(f'git -C "{Path(args.official_dir)}" rev-parse HEAD').read().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", official_commit):
        raise RuntimeError(f"Could not resolve official baseline commit: {args.official_dir}")
    metrics.update(
        {
            "mode": "v19_component_multitask_gnn",
            "method": "G-Safeguard-style GAT" if args.model_kind == "gat" else "BlindGuard-style TAM",
            "model_kind": args.model_kind,
            "official_commit": official_commit,
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": ENCODER_REVISION,
            "data_file": str((data_dir / "validation.jsonl").resolve()),
            "data_sha256": hashes["validation"],
            "dataset_role": "validation",
            "component_threshold": threshold,
            "selected_epoch": best_epoch,
            "selection_metric": "mean(validation macro-F1, validation attack-success component micro-F1@0.5)",
            "input_policy": "V19 user message only; candidate-level text and graph projection",
        }
    )
    json_dump(output_dir / "metrics.json", metrics)
    save_predictions(output_dir / "predictions.jsonl", records)
    contract = {
        "model_kind": args.model_kind,
        "official_commit": official_commit,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": ENCODER_REVISION,
        "train_sha256": hashes["train"],
        "validation_sha256": hashes["validation"],
        "test_accessed": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "localization_loss_weight": args.localization_loss_weight,
        "best_epoch": best_epoch,
        "component_threshold": threshold,
        "history": history,
        "best_model_sha256": sha256_file(output_dir / "best_model.pt"),
    }
    json_dump(output_dir / "TRAIN_CONTRACT.json", contract)
    print(json.dumps(metrics, indent=2))


def final_test(args) -> None:
    if args.sealed_test_ack != "FINAL_ONCE":
        raise RuntimeError("Final test requires --sealed-test-ack FINAL_ONCE")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError("Final-test output directory must not already exist")
    output_dir.mkdir(parents=True)
    contract = json.loads((checkpoint_dir / "TRAIN_CONTRACT.json").read_text(encoding="utf-8"))
    if contract["test_accessed"] is not False:
        raise RuntimeError("Invalid training contract")
    actual_commit = os.popen(f'git -C "{Path(args.official_dir).parent}" rev-parse HEAD').read().strip()
    if actual_commit != contract["official_commit"]:
        raise RuntimeError(
            f"Official baseline commit mismatch: {actual_commit} != {contract['official_commit']}"
        )
    test_path = Path(args.test_file)
    actual_hash = sha256_file(test_path)
    if actual_hash != MARBLE_SHA256["test"]:
        raise RuntimeError(f"Frozen MARBLE test hash mismatch: {actual_hash}")
    raw = build_raw_graphs(test_path)
    cache_contract = {
        "data_sha256": actual_hash,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": ENCODER_REVISION,
        "candidate_graph_schema": "v19-component-graph-v1",
    }
    rows = encode_graphs(raw, Path(args.cache_dir) / "test.pt", cache_contract)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    args.model_kind = contract["model_kind"]
    args.hidden_dim = contract["hidden_dim"]
    args.latent_dim = contract["latent_dim"]
    model = make_model(args, int(rows[0]["x"].shape[1]), device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))
    metrics, records = evaluate(model, rows, device, float(contract["component_threshold"]))
    metrics.update(
        {
            "mode": "v19_component_multitask_gnn",
            "method": "G-Safeguard-style GAT" if args.model_kind == "gat" else "BlindGuard-style TAM",
            "model_kind": args.model_kind,
            "official_commit": contract["official_commit"],
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": ENCODER_REVISION,
            "data_file": str(test_path.resolve()),
            "data_sha256": actual_hash,
            "dataset_role": "test",
            "component_threshold": contract["component_threshold"],
        }
    )
    json_dump(output_dir / "SEALED_TEST_CONSUMED.json", {"test_sha256": actual_hash, "rows": len(rows)})
    json_dump(output_dir / "metrics.json", metrics)
    save_predictions(output_dir / "predictions.jsonl", records)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train-validation")
    train_parser.add_argument("--model-kind", choices=["gat", "tam"], required=True)
    train_parser.add_argument("--official-dir", required=True)
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--cache-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=2e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--latent-dim", type=int, default=256)
    train_parser.add_argument("--localization-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--grad-accum", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    test_parser = subparsers.add_parser("final-test")
    test_parser.add_argument("--checkpoint-dir", required=True)
    test_parser.add_argument("--official-dir", required=True)
    test_parser.add_argument("--test-file", required=True)
    test_parser.add_argument("--cache-dir", required=True)
    test_parser.add_argument("--output-dir", required=True)
    test_parser.add_argument("--sealed-test-ack", choices=["FINAL_ONCE"], required=True)
    args = parser.parse_args()
    if args.command == "train-validation":
        train(args)
    else:
        final_test(args)


if __name__ == "__main__":
    main()
