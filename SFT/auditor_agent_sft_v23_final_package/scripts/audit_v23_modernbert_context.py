from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer


def q(values):
    values = sorted(values)
    def at(p): return values[round((len(values) - 1) * p)] if values else 0
    return {"min": values[0] if values else 0, "p50": at(.5), "p90": at(.9), "p95": at(.95), "p99": at(.99), "max": values[-1] if values else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    report = {"version": "V23-ModernBERT-context-audit-v1", "model": a.model, "revision": a.revision, "max_len": a.max_len, "input_mode": "user", "splits": {}}
    total_over = 0
    for split in ("train", "validation", "test"):
        rows = [json.loads(x) for x in (a.data_dir / f"{split}.jsonl").open(encoding="utf-8") if x.strip()]
        idx = [json.loads(x) for x in (a.data_dir / f"{split}_track_index.jsonl").open(encoding="utf-8") if x.strip()]
        if [r["metadata"]["run_id"] for r in rows] != [x["run_id"] for x in idx]: raise RuntimeError(f"{split}: index mismatch")
        lengths, over = [], []
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start:start+a.batch_size]
            ls = tok([r["messages"][1]["content"] for r in batch], add_special_tokens=True, return_length=True)["length"]
            for off, (row, ix, n) in enumerate(zip(batch, idx[start:start+len(batch)], ls)):
                lengths.append(int(n))
                if n > a.max_len:
                    over.append({"position": start+off, "run_id": ix["run_id"], "track": ix["track"], "verdict": ix["verdict"], "tokens": int(n), "origin": row.get("metadata",{}).get("dataset_version","V22")})
        total_over += len(over)
        report["splits"][split] = {"rows": len(rows), "lengths": q(lengths), "over_budget_rows": len(over), "over_budget_by_track": dict(Counter(x["track"] for x in over)), "over_budget_by_origin": dict(Counter(x["origin"] for x in over)), "over_budget": over}
    report["total_over_budget_rows"] = total_over
    report["status"] = "PASS_WITH_COMMON_SUBSET_REQUIRED" if total_over else "PASS"
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "total_over_budget_rows": total_over, "splits": {k:{x:y for x,y in v.items() if x!='over_budget'} for k,v in report["splits"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
