from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
PROMPT_VERSION = "v22-three-field-evidence-grounded-v3"
CONFIDENCE = {"high", "medium", "low"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def redact(text: str) -> str:
    text = re.sub(
        r'''(?ix)(password|access[_ -]?token|api[_ -]?key|secret)
        ["']?\s*[=:]\s*["']?[^\s,;\]}"']+''',
        r'\1: [REDACTED]', text,
    )
    text = re.sub(r'(?i)bearer\s+[a-z0-9._-]{8,}', 'Bearer [REDACTED]', text)
    text = re.sub(r'eyJ[a-zA-Z0-9._-]{20,}', '[REDACTED_TOKEN]', text)
    text = re.sub(r'(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', '[REDACTED_EMAIL]', text)
    text = re.sub(r'(?<!\d)\d{7,15}(?!\d)', '[REDACTED_NUMBER]', text)
    return text[:1600]


def evidence(row: dict, report: dict) -> list[dict]:
    user = json.loads(row["messages"][1]["content"])
    run_evidence = user.get("run_evidence", {})
    items = run_evidence.get("observed", []) + run_evidence.get("final_output", [])
    index = {str(item["id"]): item for item in items if isinstance(item, dict) and item.get("id")}
    refs = []
    for step in report["audit_trace"]:
        for ref in map(str, step.get("evidence_refs", [])):
            if ref not in refs:
                refs.append(ref)
    if not set(refs).issubset(index):
        raise ValueError(f"Invisible evidence refs: {set(refs) - set(index)}")
    output = []
    for ref in refs:
        item = index[ref]
        output.append({
            "id": ref, "type": item.get("type"), "agent": item.get("agent"),
            "tool": item.get("tool"), "text": redact(str(item.get("text", ""))),
        })
    return output


def prompt(row: dict) -> list[dict]:
    report = json.loads(row["messages"][2]["content"])
    user = json.loads(row["messages"][1]["content"])
    frozen_trace = [
        {
            "step": step["step"],
            "component_refs": step.get("component_refs", []),
            "evidence_refs": step.get("evidence_refs", []),
        }
        for step in report["audit_trace"]
    ]
    payload = {
        "run_id": row["metadata"]["run_id"],
        "task": redact(json.dumps(user.get("task", {}), ensure_ascii=False))[:1200],
        "frozen_audit": {
            "decision": {
                "verdict": report["decision"]["verdict"],
                "binary_label": report["decision"]["binary_label"],
            },
            "attack": report["attack"], "localization": report["localization"],
            "audit_trace": frozen_trace,
        },
        "visible_cited_evidence": evidence(row, report),
    }
    return [
        {"role": "system", "content": (
            "You enrich a security audit training target. The frozen audit fields and evidence IDs are authoritative. "
            "Return exactly one JSON object with only causal_explanation, recommended_action, and confidence. "
            "causal_explanation must explain the observable path from cited events to the verdict in 2-4 concise sentences "
            "and explicitly cite at least two supplied evidence IDs (or every supplied ID when fewer than two exist); "
            "do not claim more than the evidence shows. recommended_action must be one safe, component-specific defensive action "
            "and must not reveal or reproduce attack instructions or secrets. confidence must be high, medium, or low and reflect "
            "evidence sufficiency: high requires direct localization and outcome evidence, medium means partial/indirect support, "
            "and low means missing or conflicting support. Never add labels, components, credentials, personal data, "
            "or hidden facts. Never invent an evidence ID. Do not merely paraphrase event types; explain what the cited "
            "observable events establish."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def extract(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Teacher returned no JSON object")
    value = json.loads(text[start:end + 1])
    if set(value) != {"causal_explanation", "recommended_action", "confidence"}:
        raise ValueError(f"Teacher returned wrong keys: {set(value)}")
    if value["confidence"] not in CONFIDENCE:
        raise ValueError("Teacher returned invalid confidence")
    for key in ("causal_explanation", "recommended_action"):
        if not isinstance(value[key], str) or len(value[key].strip()) < 20:
            raise ValueError(f"Teacher returned weak {key}")
    serialized = json.dumps(value, ensure_ascii=False)
    if re.search(r'(?i)"(?:password|access_token|api_key|secret)"\s*:', serialized):
        raise ValueError("Teacher output contains secret-like field")
    return value


def apply_template(tokenizer, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def validate_evidence_grounding(row: dict, enrichment: dict) -> None:
    report = json.loads(row["messages"][2]["content"])
    valid_refs = {item["id"] for item in evidence(row, report)}
    cited_refs = set(re.findall(r"\b(?:obs|out)_\d+\b", enrichment["causal_explanation"]))
    required_refs = min(2, len(valid_refs))
    if len(cited_refs) < required_refs or not cited_refs.issubset(valid_refs):
        raise ValueError(
            f"Teacher causal evidence gate failed for {row['metadata']['run_id']}: "
            f"cited={sorted(cited_refs)}, valid={sorted(valid_refs)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-input-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model != "Qwen/Qwen3-32B" or args.revision != REVISION:
        raise ValueError("V22 teacher requires the pinned Qwen3-32B revision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = read(args.train_file)
    if len(rows) != 3122:
        raise RuntimeError("V22 teacher requires the frozen 3122-row train split")
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(rows):
            raise ValueError("--limit must be between 1 and 3122")
        rows = rows[:args.limit]
    completed = read(args.output) if args.output.is_file() else []
    if len(completed) > len(rows):
        raise RuntimeError("Teacher resume output is longer than train data")
    for index, item in enumerate(completed):
        if item["run_id"] != rows[index]["metadata"]["run_id"]:
            raise RuntimeError(f"Teacher resume run-id prefix mismatch at {index}")
        if item.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError("Teacher resume output uses an older prompt contract; use a fresh output path")
        extract(json.dumps(item["enrichment"], ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.output.open(mode, encoding="utf-8") as writer:
        progress = tqdm(total=len(rows), initial=len(completed), desc="qwen32b_v22_teacher")
        remaining = rows[len(completed):]
        for start in range(0, len(remaining), args.batch_size):
            batch_rows = remaining[start:start + args.batch_size]
            texts = [apply_template(tokenizer, prompt(row)) for row in batch_rows]
            encoded = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > args.max_input_len:
                raise ValueError(f"Teacher zero-truncation gate failed: {encoded['input_ids'].shape[1]}")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            suffix = tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch_rows, suffix):
                last_error = None
                for attempt in range(3):
                    try:
                        enrichment = extract(text)
                        validate_evidence_grounding(row, enrichment)
                        break
                    except (ValueError, json.JSONDecodeError) as error:
                        last_error = error
                        if attempt == 2:
                            raise RuntimeError(
                                f"Teacher repair failed after 3 attempts for {row['metadata']['run_id']}"
                            ) from last_error
                        repair_messages = prompt(row) + [
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": (
                                "Correct the JSON. The causal_explanation must explicitly cite at least two evidence IDs "
                                "already supplied in visible_cited_evidence (or all IDs if fewer than two). Do not invent IDs, "
                                "change the frozen verdict/localization, or add fields. Return only the corrected JSON object."
                            )},
                        ]
                        repair_text = apply_template(tokenizer, repair_messages)
                        repair_encoded = tokenizer(
                            repair_text, add_special_tokens=False, return_tensors="pt"
                        )
                        if repair_encoded["input_ids"].shape[1] > args.max_input_len:
                            raise ValueError("Teacher repair zero-truncation gate failed")
                        repair_encoded = {key: value.to(model.device) for key, value in repair_encoded.items()}
                        with torch.no_grad():
                            repair_output = model.generate(
                                **repair_encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        text = tokenizer.decode(
                            repair_output[0, repair_encoded["input_ids"].shape[1]:], skip_special_tokens=True
                        )
                item = {
                    "run_id": row["metadata"]["run_id"],
                    "prompt_version": PROMPT_VERSION,
                    "enrichment": enrichment,
                }
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.flush()
                progress.update(1)
        progress.close()
    contract = {
        "rows": len(rows), "source_rows": 3122, "train_sha256": sha256(args.train_file),
        "output_sha256": sha256(args.output),
        "model": args.model, "revision": args.revision, "load_in_4bit": True,
        "fields": ["causal_explanation", "recommended_action", "confidence"],
        "prompt_version": PROMPT_VERSION,
        "validation_gold_accessed": False, "sealed_test_accessed": False,
    }
    args.output.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
