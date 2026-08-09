from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def signature(value, prefix=""):
    out = []
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else key
            out.append((path, "dict-key"))
            out.extend(signature(value[key], path))
    elif isinstance(value, list):
        out.append((prefix, "list"))
        for item in value[:5]:
            out.extend(signature(item, prefix + "[]"))
    else:
        out.append((prefix, type(value).__name__))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"status": "PASS", "tracks": {}, "differences": {}}
    signatures = {}
    for track in ("marble_only", "autogen_only", "mixed"):
        counter = Counter()
        schemas = Counter()
        for split in ("train", "validation", "test"):
            with (args.track_root / track / f"{split}.jsonl").open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    user = json.loads(row["messages"][1]["content"])
                    assistant = json.loads(row["messages"][2]["content"])
                    schemas[user.get("schema")] += 1
                    counter.update(signature({"row": row, "user": user, "assistant": assistant}))
        signatures[track] = set(counter)
        result["tracks"][track] = {"schemas": dict(schemas), "signature_items": len(counter)}
    reference = signatures["marble_only"]
    for track in ("autogen_only", "mixed"):
        missing = sorted(reference - signatures[track])
        extra = sorted(signatures[track] - reference)
        result["differences"][track] = {"missing": missing, "extra": extra}
        if missing or extra:
            result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
