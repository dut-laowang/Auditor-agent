#!/usr/bin/env python3
"""Evaluate label-blind exhaustive and deterministic selective verifiers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from score_predictions_by_track import metrics

def read(p): return [json.loads(x) for x in Path(p).open(encoding="utf-8") if x.strip()]
def write(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def confidence(row):
    try:return str(json.loads(row.get("generation") or "{}").get("decision",{}).get("confidence","missing")).lower()
    except (ValueError,TypeError):return "parse_error"
def main():
    p=argparse.ArgumentParser();p.add_argument("--qwen",required=True);p.add_argument("--bert",required=True);p.add_argument("--index",required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--max-verify-rate",type=float,default=.15);a=p.parse_args()
    q={r["run_id"]:r for r in read(a.qwen)};b={r["run_id"]:r for r in read(a.bert)};idx=read(a.index);ids=[r["run_id"] for r in idx]
    if len(ids)!=len(set(ids)) or set(ids)-set(q) or set(ids)-set(b):raise RuntimeError("component-policy ID contract mismatch")
    fixed=[dict(b[i]) for i in ids]
    priorities=[]
    rank={"parse_error":0,"missing":1,"low":2,"medium":3,"high":4}
    for pos,i in enumerate(ids):
        invalid=not bool((q[i].get("trace_quality") or {}).get("valid_json",True));disagree=q[i]["pred"]!=b[i]["pred"]
        priorities.append(((0 if invalid else 1),(0 if disagree else 1),rank.get(confidence(q[i]),2),pos,i))
    budget=min(len(ids),int(len(ids)*a.max_verify_rate));chosen={x[-1] for x in sorted(priorities)[:budget]}
    rule=[];corrected=corrupted=0
    for i in ids:
        use=i in chosen and q[i]["pred"]!=b[i]["pred"]; row=dict(b[i] if use else q[i]);rule.append(row)
        if use:
            corrected+=int(q[i]["pred"]!=q[i]["gold"] and b[i]["pred"]==b[i]["gold"])
            corrupted+=int(q[i]["pred"]==q[i]["gold"] and b[i]["pred"]!=b[i]["gold"])
    for name,rows,extra in (("fixed_cascade",fixed,{"verify_rows":len(ids),"verify_rate":1.0}), ("rule_router",rule,{"verify_rows":len(chosen),"verify_rate":len(chosen)/len(ids),"corrected":corrected,"corrupted":corrupted})):
        out=a.output_dir/name;write(out/"predictions.jsonl",rows);m=metrics(rows);m.update(extra);m.update({"coverage":1.0,"defer_rate":0.0,"label_blind_test_policy":True,"policy_version":"V23-component-policy-v1"});(out/"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
    (a.output_dir/"COMPONENT_POLICIES_COMPLETE.json").write_text(json.dumps({"status":"PASS","rows":len(ids),"rule_budget":budget},indent=2),encoding="utf-8")
if __name__=="__main__":main()
