from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
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
SCOPES = ("none", "global", "node", "edge", "tool", "multi")
SCOPE_TO_ID = {name: idx for idx, name in enumerate(SCOPES)}
TYPE_ORDER = ("global", "node", "edge", "tool", "unknown")
MARBLE_SHA256 = {
    "train": "d49ec56577a80fab3be360ae9cb1b90e2d751ce686ffa0f0e2064b2d05d0a932",
    "validation": "2882c5adfe2b9c3e7820f7cf56338dec4bf59e091031763ce8ce46d6aa70609e",
    "test": "bee77d962f66f5481e88d89b49b83b3ea9a449e48d776b669ebadd731417167f",
}
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
OFFICIAL_COMMITS = {
    "gat": "890c99f1cbc864e9ff0c85859619a14f42bc9cab",
    "tam": "1889c20a326ba9ba9a6982744d473626e74f9986",
}
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


def cpu_cuda_rng_states(states) -> list[torch.Tensor]:
    """PyTorch requires CUDA RNG restore states to be CPU uint8 tensors."""
    converted = [state.detach().to(device="cpu", dtype=torch.uint8) for state in states]
    if any(state.device.type != "cpu" or state.dtype != torch.uint8 for state in converted):
        raise RuntimeError("Invalid CUDA RNG checkpoint state")
    return converted


def official_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", result):
        raise RuntimeError(f"Could not resolve official baseline commit: {path}")
    return result


def official_source_identity(kind: str, path: Path) -> dict:
    source = path / ("model.py" if kind == "gat" else "TAM.py")
    if not source.is_file():
        raise RuntimeError(f"Missing official {kind} source: {source}")
    return {"path": source.name, "sha256": sha256_file(source)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def deterministic_stratified_limit(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    groups = {name: [] for name in VERDICTS}
    for row in rows:
        groups[row["gold_verdict"]].append(row)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for name in VERDICTS:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
    if len(selected) != min(limit, len(rows)):
        raise RuntimeError("Could not construct deterministic stratified limit")
    return selected


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
            for key in ("source", "type", "agent", "tool", "source_agent", "target_agent")
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
                for key in ("source", "type", "agent", "tool", "source_agent", "target_agent")
                if event.get(key) is not None
            )
            pieces.append((header + " " + str(event.get("text") or "")).strip())
        graph = user.get("graph", {})
        coverage = user.get("run_evidence", {}).get("coverage", {})
        pieces.append(
            "graph_context="
            + json.dumps(
                {
                    "topology": graph.get("topology"),
                    "audit_mode": user.get("audit_request", {}).get("mode"),
                    "coverage": coverage,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
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
    tokenizer = encoder.tokenizer
    token_capacity = int(encoder.max_seq_length) - int(tokenizer.num_special_tokens_to_add(pair=False))
    if token_capacity <= 0:
        raise RuntimeError("Invalid sentence-encoder token capacity")
    chunk_ids = []
    chunk_owners = []
    chunk_weights = []
    max_piece_tokens = 0
    chunked_pieces = 0
    tokenizer_limit = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    for owner, piece in enumerate(tqdm(unique_pieces, desc="zero_truncation_tokenize")):
        ids = tokenizer.encode(piece, add_special_tokens=False, truncation=False)
        max_piece_tokens = max(max_piece_tokens, len(ids))
        chunks = [ids[offset : offset + token_capacity] for offset in range(0, len(ids), token_capacity)] or [[]]
        chunked_pieces += len(chunks) > 1
        for chunk in chunks:
            chunk_ids.append(chunk)
            chunk_owners.append(owner)
            chunk_weights.append(max(len(chunk), 1))
    tokenizer.model_max_length = tokenizer_limit
    embedding_dim = int(encoder.get_sentence_embedding_dimension())
    weighted_sums = np.zeros((len(unique_pieces), embedding_dim), dtype=np.float64)
    weight_sums = np.zeros(len(unique_pieces), dtype=np.float64)
    encoder.eval()
    for start in tqdm(range(0, len(chunk_ids), 256), desc="zero_truncation_encode"):
        stop = min(start + 256, len(chunk_ids))
        prepared = []
        for ids in chunk_ids[start:stop]:
            input_ids = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
            prepared.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "token_type_ids": [0] * len(input_ids),
                }
            )
        features = tokenizer.pad(prepared, padding=True, return_tensors="pt")
        features = {key: value.to(encoder.device) for key, value in features.items()}
        with torch.no_grad():
            batch_embeddings = F.normalize(encoder(features)["sentence_embedding"], p=2, dim=1)
        for local_index, vector in enumerate(batch_embeddings.cpu().numpy()):
            chunk_index = start + local_index
            owner = chunk_owners[chunk_index]
            weight = chunk_weights[chunk_index]
            weighted_sums[owner] += vector.astype(np.float64) * weight
            weight_sums[owner] += weight
    embeddings = weighted_sums / weight_sums[:, None]
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    embeddings = embeddings.astype(np.float32)
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
    json_dump(
        cache_path.with_suffix(".encoding_audit.json"),
        {
            "status": "PASS",
            "zero_truncation": True,
            "unique_pieces": len(unique_pieces),
            "encoded_chunks": len(chunk_ids),
            "chunked_pieces": chunked_pieces,
            "max_piece_tokens": max_piece_tokens,
            "token_capacity_per_chunk": token_capacity,
        },
    )
    return encoded


def graph_tensors(row: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.tensor(row["x"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(row["edge_index"], dtype=torch.long, device=device)
    verdict = torch.tensor(VERDICT_TO_ID[row["gold_verdict"]], dtype=torch.long, device=device)
    if row["gold_scope"] not in SCOPE_TO_ID:
        raise ValueError(f"Unknown localization scope: {row['gold_scope']}")
    scope = torch.tensor(SCOPE_TO_ID[row["gold_scope"]], dtype=torch.long, device=device)
    gold = set(row["gold_components"])
    components = torch.tensor(
        [1.0 if cid in gold else 0.0 for cid in row["candidate_ids"]],
        dtype=torch.float32,
        device=device,
    )
    return x, edge_index, verdict, scope, components


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
    """TAM encoder used only as a supervised V20 adaptation, not BlindGuard."""

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
        self.scope_head = nn.Linear(latent_dim * 2, len(SCOPES))
        self.verdict_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 3)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        node_embeddings = self.encoder(x, edge_index)
        pooled = torch.cat([node_embeddings.mean(dim=0), node_embeddings.max(dim=0).values])
        return (
            self.verdict_head(pooled),
            self.scope_head(pooled),
            self.loc_head(node_embeddings).squeeze(-1),
        )


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
    by_scope = {}
    for gold_scope in ("global", "node", "edge", "tool", "multi"):
        scoped = [row for row in rows if row["gold_scope"] == gold_scope]
        if not scoped:
            continue
        by_scope[gold_scope] = {
            "n": len(scoped),
            "component_hit_rate": safe_div(
                sum(bool(set(row["gold_components"]) & set(row["pred_components"])) for row in scoped),
                len(scoped),
            ),
            "component_exact_match": safe_div(
                sum(set(row["gold_components"]) == set(row["pred_components"]) for row in scoped),
                len(scoped),
            ),
            "scope_accuracy": safe_div(
                sum(row["gold_scope"] == row["pred_scope"] for row in scoped), len(scoped)
            ),
            "predicted_scope_distribution": dict(Counter(row["pred_scope"] for row in scoped)),
        }
    return {
        "n": len(rows),
        "component_micro_precision": precision,
        "component_micro_recall": recall,
        "component_micro_f1": f1,
        "component_hit_rate": safe_div(hit, len(rows)),
        "component_exact_match": safe_div(exact, len(rows)),
        "scope_accuracy": safe_div(scope, len(rows)),
        "by_gold_scope": by_scope,
    }


def evaluate(model, rows: list[dict], device: torch.device, threshold: float) -> tuple[dict, list[dict]]:
    records = []
    model.eval()
    with torch.no_grad():
        for row in tqdm(rows, desc="evaluate"):
            x, edge_index, _, _, _ = graph_tensors(row, device)
            verdict_logits, scope_logits, component_logits = model(x, edge_index)
            pred_verdict = VERDICTS[int(verdict_logits.argmax().item())]
            pred_scope = SCOPES[int(scope_logits.argmax().item())]
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
                    "pred_scope": pred_scope,
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
    resolved_official_commit = official_commit(Path(args.official_dir))
    source_identity = official_source_identity(args.model_kind, Path(args.official_dir))
    if resolved_official_commit != OFFICIAL_COMMITS[args.model_kind]:
        raise RuntimeError(
            f"Wrong official {args.model_kind} commit: {resolved_official_commit} != "
            f"{OFFICIAL_COMMITS[args.model_kind]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    encoded = {}
    expected_hashes = {
        "train": args.expected_train_sha256,
        "validation": args.expected_validation_sha256,
    }
    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        actual = sha256_file(path)
        if actual != expected_hashes[split]:
            raise RuntimeError(
                f"Frozen dataset {split} hash mismatch: {actual} != {expected_hashes[split]}"
            )
        hashes[split] = actual
        raw = build_raw_graphs(path)
        if args.limit is not None:
            raw = deterministic_stratified_limit(raw, args.limit)
            if not raw:
                raise RuntimeError("--limit produced an empty split")
        contract = {
            "data_sha256": actual,
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": ENCODER_REVISION,
            "candidate_graph_schema": "v19-component-graph-v2-zero-truncation",
            "limit": args.limit,
        }
        suffix = f"_limit{args.limit}" if args.limit is not None else ""
        encoded[split] = encode_graphs(raw, cache_dir / f"{split}{suffix}.pt", contract)
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
    resume_contract = {
        "model_kind": args.model_kind,
        "official_commit": resolved_official_commit,
        "official_source_identity": source_identity,
        "train_sha256": hashes["train"],
        "validation_sha256": hashes["validation"],
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "localization_loss_weight": args.localization_loss_weight,
        "scope_loss_weight": args.scope_loss_weight,
        "limit": args.limit,
    }
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("resume_contract") != resume_contract:
            raise RuntimeError("Existing GNN checkpoint contract does not match this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_score = float(state["best_score"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(cpu_cuda_rng_states(state["cuda_rng_state_all"]))
    history = list(state.get("history", [])) if last_path.is_file() else []
    for epoch in range(start_epoch, args.epochs + 1):
        order = list(range(len(encoded["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(tqdm(order, desc=f"{args.model_kind}_epoch_{epoch}"), 1):
            x, edge_index, verdict, scope, components = graph_tensors(encoded["train"][index], device)
            verdict_logits, scope_logits, component_logits = model(x, edge_index)
            verdict_loss = F.cross_entropy(verdict_logits.unsqueeze(0), verdict.unsqueeze(0), weight=class_weights)
            scope_loss = F.cross_entropy(scope_logits.unsqueeze(0), scope.unsqueeze(0))
            component_loss = F.binary_cross_entropy_with_logits(component_logits, components, pos_weight=pos_weight)
            loss = (
                verdict_loss
                + args.scope_loss_weight * scope_loss
                + args.localization_loss_weight * component_loss
            ) / args.grad_accum
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
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "resume_contract": resume_contract,
            },
            last_path,
        )
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    threshold = select_threshold(model, encoded["validation"], device)
    metrics, records = evaluate(model, encoded["validation"], device, threshold)
    official_commit_hash = resolved_official_commit
    metrics.update(
        {
            "mode": "v19_component_multitask_gnn",
            "method": "G-Safeguard" if args.model_kind == "gat" else "TAM encoder (supervised adaptation)",
            "model_kind": args.model_kind,
            "official_commit": official_commit_hash,
            "official_source_identity": source_identity,
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
        "official_commit": official_commit_hash,
        "official_source_identity": source_identity,
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
        "scope_loss_weight": args.scope_loss_weight,
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Non-empty final-test output directory already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads((checkpoint_dir / "TRAIN_CONTRACT.json").read_text(encoding="utf-8"))
    if contract["test_accessed"] is not False:
        raise RuntimeError("Invalid training contract")
    actual_commit = official_commit(Path(args.official_dir))
    if actual_commit != contract["official_commit"]:
        raise RuntimeError(
            f"Official baseline commit mismatch: {actual_commit} != {contract['official_commit']}"
        )
    actual_source_identity = official_source_identity(contract["model_kind"], Path(args.official_dir))
    if actual_source_identity != contract["official_source_identity"]:
        raise RuntimeError("Official baseline source identity mismatch")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != contract["best_model_sha256"]:
        raise RuntimeError(
            f"Best-model hash mismatch: {checkpoint_hash} != {contract['best_model_sha256']}"
        )
    test_path = Path(args.test_file)
    actual_hash = sha256_file(test_path)
    if actual_hash != args.expected_test_sha256:
        raise RuntimeError(f"Frozen test hash mismatch: {actual_hash} != {args.expected_test_sha256}")
    raw = build_raw_graphs(test_path)
    if args.limit is not None:
        raw = deterministic_stratified_limit(raw, args.limit)
        if not raw:
            raise RuntimeError("--limit produced an empty test split")
    cache_contract = {
        "data_sha256": actual_hash,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": ENCODER_REVISION,
        "candidate_graph_schema": "v19-component-graph-v2-zero-truncation",
        "limit": args.limit,
    }
    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    rows = encode_graphs(raw, Path(args.cache_dir) / f"test{suffix}.pt", cache_contract)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    args.model_kind = contract["model_kind"]
    args.hidden_dim = contract["hidden_dim"]
    args.latent_dim = contract["latent_dim"]
    model = make_model(args, int(rows[0]["x"].shape[1]), device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    metrics, records = evaluate(model, rows, device, float(contract["component_threshold"]))
    metrics.update(
        {
            "mode": "v19_component_multitask_gnn",
            "method": "G-Safeguard" if args.model_kind == "gat" else "TAM encoder (supervised adaptation)",
            "model_kind": args.model_kind,
            "official_commit": contract["official_commit"],
            "official_source_identity": actual_source_identity,
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": ENCODER_REVISION,
            "data_file": str(test_path.resolve()),
            "data_sha256": actual_hash,
            "dataset_role": "test",
            "component_threshold": contract["component_threshold"],
            "checkpoint_sha256": checkpoint_hash,
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
    train_parser.add_argument("--scope-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--grad-accum", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--limit", type=int)
    train_parser.add_argument("--expected-train-sha256", default=MARBLE_SHA256["train"])
    train_parser.add_argument(
        "--expected-validation-sha256", default=MARBLE_SHA256["validation"]
    )
    test_parser = subparsers.add_parser("final-test")
    test_parser.add_argument("--checkpoint-dir", required=True)
    test_parser.add_argument("--official-dir", required=True)
    test_parser.add_argument("--test-file", required=True)
    test_parser.add_argument("--cache-dir", required=True)
    test_parser.add_argument("--output-dir", required=True)
    test_parser.add_argument("--sealed-test-ack", choices=["FINAL_ONCE"], required=True)
    test_parser.add_argument("--expected-test-sha256", default=MARBLE_SHA256["test"])
    test_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "train-validation":
        train(args)
    else:
        final_test(args)


if __name__ == "__main__":
    main()
