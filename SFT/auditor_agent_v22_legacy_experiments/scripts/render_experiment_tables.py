#!/usr/bin/env python3
"""Render the five frozen-protocol publication tables; never impute a result."""
from __future__ import annotations
import argparse, json
from pathlib import Path

METRICS=("three_class_accuracy","three_class_macro_f1","attack_success_recall","binary_accuracy","localization_micro_f1")
V18=[
 ["Single-agent guardrail","AgentDoG Frozen Official","46.50","30.40","58.60","N/A","N/A","N/A","N/A","200"],
 ["Single-agent guardrail","AgentDoG Frozen + adapted prompt","60.00","35.90","48.30","N/A","N/A","N/A","N/A","200"],
 ["Domain-adapted transfer","AgentDoG + V18-Flat SFT","82.50","71.70","65.50","74.00","73.70","66.70","70.70","200"],
 ["Domain-adapted transfer","Qwen V18-Flat","82.50","74.50","60.30","74.50","74.00","61.30","60.30","200"],
 ["Graph transfer","Qwen V18-Graph","81.50","73.30","56.90","74.50","73.80","61.30","67.20","200"],
 ["Online MAS audit","AgentForesight (official paper)","N/A","N/A","N/A","N/A","N/A","N/A","N/A","official: Exact-F1 66.44; ASS 0.59; FAR 2.37%"],
]
V19=[
 ["Clean","75.60","74.95","58.78","73.61","82.86","56.12","64.69","—","—"],
 ["+ Cross-label text rotation","13.46","12.41","6.53","16.48","15.31","7.55","11.84","−62.14","−57.13"],
 ["− Event text","43.27","20.14","0.00","15.36","17.76","8.37","22.45","−32.33","−58.25"],
 ["− Lexical shortcuts","48.52","34.79","38.57","59.65","76.94","38.98","47.55","−27.08","−13.96"],
 ["+ Event shuffling","74.65","73.18","48.78","55.11","65.71","41.63","54.08","−0.95","−18.50"],
 ["− Task goal","75.82","75.20","59.18","73.02","82.65","55.31","64.08","+0.22","−0.59"],
 ["− Structure links","75.71","75.03","58.57","73.36","82.86","55.71","64.08","+0.11","−0.25"],
 ["− Outcome text","75.66","74.93","57.76","73.85","83.06","56.73","64.90","+0.06","+0.24"],
]
def load(p):
 try:return json.loads(Path(p).read_text(encoding="utf-8"))
 except (FileNotFoundError,json.JSONDecodeError):return None
def first(*ps):
 return next((x for p in ps if (x:=load(p)) is not None),None)
def deep(d,k):
 if not d:return None
 if k in d:return d[k]
 if isinstance(d.get("agent_final_full_coverage"),dict):
  nested=deep(d["agent_final_full_coverage"],k)
  if nested is not None:return nested
 for q in ("summary","metrics","test","delta","agent_final_full_coverage"):
  if isinstance(d.get(q),dict) and k in d[q]:return d[q][k]
 if k=="three_class_macro_f1":return d.get("three_class_report",{}).get("macro avg",{}).get("f1-score")
 if k=="localization_micro_f1":return d.get("localization",{}).get("component_micro_f1")
 if k=="attack_success_recall":return d.get("three_class_report",{}).get("attack_success",{}).get("recall")
def pct(d,k):
 v=deep(d,k); return "TBD" if v is None else f"{100*float(v):.2f}"
def val(d,k,scale=100):
 v=deep(d,k); return "TBD" if v is None else f"{scale*float(v):.2f}"
def nval(d,default="TBD"):
 if not d:return default
 return str(deep(d,"n") or deep(d,"rows") or default)
def mdtable(title,headers,rows):
 z=[f"## {title}","", "| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |"]
 z += ["| "+" | ".join(map(str,r))+" |" for r in rows]; return "\n".join(z)
def latex_escape(s):
 s=str(s)
 for a,b in (("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("_",r"\_"),("#",r"\#")):s=s.replace(a,b)
 return s
def latex_table(title,headers,rows):
 newline = " " + chr(92) * 2
 lines=[r"\begin{table*}[t]",r"\centering",r"\scriptsize",f"\\caption{{{latex_escape(title)}}}","\\begin{tabular}{l"+"r"*(len(headers)-1)+"}",r"\toprule"," & ".join(map(latex_escape,headers))+newline,r"\midrule"]
 last=None
 for r in rows:
  if last is not None and r[0]!=last:lines.append(r"\midrule")
  lines.append(" & ".join(map(latex_escape,r))+newline);last=r[0]
 return "\n".join(lines+[r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run-dir",type=Path,required=True);ap.add_argument("--supplement-dir",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args();r,s=a.run_dir,a.supplement_dir
 def get(*rels):return first(*[root/rel for rel in rels for root in (r,s)])
 q=get("qwen3_8b_plain_sft_test/metrics.json","qwen3_8b_plain_sft_test/test_metrics.json")
 b=get("modernbert_sealed_test/metrics.json","modernbert_sealed_test/test_metrics.json","modernbert_eval/test/metrics.json")
 ag=get("plain_hetero_agent_full_test_common2531_v3/AGENT_TEST_COMPARISON.json","plain_hetero_agent_full_test/AGENT_TEST_COMPARISON.json","plain_hetero_agent_full_test/AGENT_FULL_TEST_COMPLETE.json","plain_hetero_agent_full_test/summary.json","plain_hetero_agent_full_test/metrics.json")
 defs=[
 ("Open-source general LLM","Qwen3-8B Base",get("baselines/qwen3_8b_base_ztr12288_v2/metrics.json","baselines/qwen3_8b_base/metrics.json","baselines/qwen3_8b_base_common200/metrics.json"),"Zero-shot fixed prompt; zero truncation"),
 ("Open-source general LLM","Qwen3-32B Base",get("baselines/qwen3_32b_base_ztr12288_v2/metrics.json","baselines/qwen3_32b_base/metrics.json","baselines/qwen3_32b_base_common200/metrics.json"),"Zero-shot fixed prompt; zero truncation"),
 ("Closed-source general LLM","GPT-4.1",get("baselines/gpt41/metrics.json","baselines/gpt41_common200/metrics.json"),"Zero-shot fixed prompt"),
 ("Graph MAS defense","G-Safeguard",get("baselines/gsafeguard_official_v22_v1/test/metrics.json"),"official MyGAT encoder; V22 supervised heads"),
 ("Graph MAS defense","TAM Encoder",get("baselines/tam_official_v22_v1/test/metrics.json"),"official TAM encoder; V22 supervised heads"),
 ("Graph MAS defense","BlindGuard",get("baselines/blindguard/metrics.json"),"V22-adapted, normal-only"),
 ("Graph MAS defense","XG-Guard (ACL'26)",get("baselines/xgguard_official_v22_v1/full/metrics.json"),"official OursMethod; V22 adapter; normal-only"),
 ("Discriminative auditor","ModernBERT",b,"V22 supervised"),
 ("Generative auditor","Plain Qwen3-8B SFT",q,"V22 audit SFT"),
 ("Our full method","Qwen SFT + BERT Bounded Agent",ag,"learned routing + selective verification")]
 rows=[]
 for cat,name,d,adapt in defs:rows.append([cat,name,adapt,*[pct(d,k) for k in METRICS],nval(d,"2,531 / 2,539" if name=="ModernBERT" else "TBD")])
 tables=[("Table 1. V22-ALL unified test results",["Category","Method","Training/adaptation","3-way Acc.","Macro-F1","AS Recall","Binary Acc.","Loc. F1","N"],rows)]
 agent_rows=[]
 for name,d in [("ModernBERT-only",b),("Plain Qwen3-8B SFT",q),("Qwen + BERT fixed cascade",get("qwen_ready_audit_sft/metrics.json")),("Rule-router pilot",get("plain_hetero_agent_test300_rule/summary.json")),("Bounded Agent (learned router)",ag)]:
  agent_rows.append([name,pct(d,"three_class_macro_f1"),pct(d,"localization_micro_f1"),val(d,"verify_rate"),val(d,"defer_rate"),val(d,"coverage"),str(deep(d,"corrected") if deep(d,"corrected") is not None else "TBD"),str(deep(d,"corrupted") if deep(d,"corrupted") is not None else "TBD"),str(deep(d,"verify_rows") if deep(d,"verify_rows") is not None else "TBD")])
 tables.append(("Table 2. Core modules, Agent utility, and cost",["Method","Final Macro-F1","Final Loc. F1","Verify %↓","Defer %↓","Coverage %↑","Corrected↑","Corrupted↓","Extra calls↓"],agent_rows))
 tables.append(("Table 3. Single-Agent-to-MAS transfer and online-audit reference",["Protocol","Method","Binary Acc.","Unsafe P","Unsafe R","3-way Acc.","Macro-F1","Loc. F1","Scope Acc.","N / native metrics"],V18))
 held=[]
 labels=[("Topology","Tree","topology__tree"),("Attack surface","Message","surface__message"),("Scenario/task","Research","scenario__research")]
 for dim,v,fold in labels:
  for method,kind in (("ModernBERT","true held-out retraining"),("Qwen3-8B SFT","frozen OOD slice"),("Bounded Agent","frozen OOD slice")):
   key="modernbert" if method=="ModernBERT" else ("qwen_frozen" if method.startswith("Qwen") else "agent_frozen")
   if method=="ModernBERT":
    d=load(s/"heldout"/fold/"modernbert_ztr_v2"/"metrics.json") or load(s/"heldout"/fold/"modernbert"/"metrics.json")
   else:
    d=load(s/"heldout"/fold/key/"metrics.json") or (load(s/"heldout"/fold/"qwen"/"metrics.json") if method.startswith("Qwen") else None)
   held.append([dim,v,method,kind,*[pct(d,k) for k in METRICS],nval(d)])
 tables.append(("Table 4. V22-ALL held-out and frozen-OOD generalization",["Dimension","Unseen value","Method","Protocol","Acc.","Macro-F1","AS Recall","Binary Acc.","Loc. F1","N"],held))
 tables.append(("Table 5. Evidence dependence and counterfactual ablations (frozen V19 validation)",["Condition","3-way Acc.","Macro-F1","AS Recall","Loc. F1","Hit","Exact","Scope Acc.","Δ Class.","Δ Loc."],V19))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 intro="# Final five experiment tables\n\n`TBD` denotes an experiment not completed under the stated protocol; `N/A` denotes a non-equivalent metric. No value is imputed.\n\n"
 md=intro+"\n\n".join(mdtable(*t) for t in tables)+"\n"
 tex="% Requires \\usepackage{booktabs}\n\n"+"\n\n".join(latex_table(*t) for t in tables)+"\n"
 (a.output_dir/"FINAL_FIVE_TABLES.md").write_text(md,encoding="utf-8");(a.output_dir/"FINAL_FIVE_TABLES.tex").write_text(tex,encoding="utf-8")
 status={"status":"PASS" if "TBD" not in md else "INCOMPLETE","tbd_cells":md.count("TBD"),"tables":5,"no_imputation":True,"protocol":"V22-ALL legacy 15931; V19 ablation and V18 transfer explicitly frozen"}
 (a.output_dir/"TABLE_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8");print(json.dumps(status,indent=2));print(a.output_dir/"FINAL_FIVE_TABLES.md")
if __name__=="__main__":main()
