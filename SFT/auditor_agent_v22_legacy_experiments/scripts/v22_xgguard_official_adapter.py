#!/usr/bin/env python3
"""V22 adapter that executes model definitions extracted from official XG-Guard Ours.py."""
from __future__ import annotations
import argparse, ast, collections, hashlib, json, math, os, random, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
from tqdm import tqdm

OFFICIAL_COMMIT="86e1121512f76800f80d4687e492c7f99f049929"
ENCODER="sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
VERDICTS=("clean_safe","attack_failed","attack_success")

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
def write_json(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf8")
def write_jsonl(path,rows):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf8")
def graph_stats(rows):
 counts=collections.Counter(len(x["nodes"]) for x in rows)
 return {"graphs":len(rows),"single_node_graphs":counts.get(1,0),"min_nodes":min(counts),"max_nodes":max(counts),"node_count_distribution":{str(k):v for k,v in sorted(counts.items())}}

def official_model(repo:Path):
 from torch_geometric.nn import GCNConv
 head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
 if head!=OFFICIAL_COMMIT:raise RuntimeError(f"XG-Guard commit mismatch: {head} != {OFFICIAL_COMMIT}")
 source=repo.joinpath("Ours.py"); tree=ast.parse(source.read_text(encoding="utf8"),filename=str(source))
 wanted={"GCNEncoder","OursMethod","get_score_overall"}
 nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in wanted]
 found={x.name for x in nodes}
 if found!=wanted:raise RuntimeError(f"Official definitions missing: {wanted-found}")
 module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
 ns={"torch":torch,"nn":nn,"F":F,"GCNConv":GCNConv}
 exec(compile(module,str(source),"exec"),ns)
 return ns["OursMethod"],ns["get_score_overall"],{"commit":head,"ours_py_sha256":sha(source),"definitions":sorted(wanted)}

def convert(row):
 u=json.loads(row["messages"][1]["content"]); gold=json.loads(row["messages"][2]["content"])
 nodes=[str(x) for x in u["graph"]["nodes"]]; pos={x:i for i,x in enumerate(nodes)}
 if not nodes or len(pos)!=len(nodes):raise ValueError("missing/duplicate graph nodes")
 texts={x:[] for x in nodes}
 events=u.get("run_evidence",{}).get("observed",[])+u.get("run_evidence",{}).get("final_output",[])
 for e in events:
  owner=next((str(e[k]) for k in ("source_agent","agent","source") if e.get(k) is not None and str(e[k]) in pos),None)
  if owner:texts[owner].append(" ".join(str(e.get(k,"")) for k in ("type","text") if e.get(k)))
 candidates={str(c["id"]):c for c in u.get("graph_candidates",[]) if isinstance(c,dict) and c.get("id")}
 for n in nodes:
  if not texts[n]:texts[n]=[json.dumps(candidates.get(f"N::{n}",{"agent":n}),ensure_ascii=False,sort_keys=True)]
 edges=[]
 for e in u["graph"].get("edges",[]):
  if str(e.get("source")) in pos and str(e.get("target")) in pos:edges.append((pos[str(e["source"])],pos[str(e["target"])]))
 edges += [(i,i) for i in range(len(nodes))]
 attack_agents=set()
 for cid in gold.get("localization",{}).get("component_ids",[]) or []:
  c=candidates.get(str(cid),{}); typ=str(c.get("type","")).lower()
  if typ=="node" and str(c.get("agent")) in pos:attack_agents.add(str(c["agent"]))
  elif typ=="edge" and str(c.get("source")) in pos:attack_agents.add(str(c["source"]))
  elif typ=="tool" and str(c.get("agent")) in pos:attack_agents.add(str(c["agent"]))
 return {"run_id":row["metadata"]["run_id"],"nodes":nodes,"node_texts":["\n".join(texts[n]) for n in nodes],"edge_index":np.asarray(sorted(set(edges)),dtype=np.int64).T,"candidates":list(candidates.values()),"gold_agents":sorted(attack_agents),"gold_verdict":gold["decision"]["verdict"],"gold_scope":gold.get("localization",{}).get("scope","none"),"gold_components":[str(x) for x in gold.get("localization",{}).get("component_ids",[]) or []],"task_summary":json.dumps(u.get("task",{}),ensure_ascii=False,sort_keys=True)}

def encode(rows,cache,source_hash):
 contract={"schema":"official-xgguard-v22-agent-graph-v1","source_sha256":source_hash,"encoder":ENCODER,"revision":ENCODER_REVISION}
 cp=Path(str(cache)+".contract.json")
 if Path(cache).is_file() and cp.is_file() and json.loads(cp.read_text()).get("identity")==contract:return torch.load(cache,map_location="cpu",weights_only=False)
 from sentence_transformers import SentenceTransformer
 enc=SentenceTransformer(ENCODER,revision=ENCODER_REVISION,cache_folder=os.environ.get("HF_HOME"),device="cuda" if torch.cuda.is_available() else "cpu")
 out=[]; truncated=0
 for g in tqdm(rows,desc="official_xgguard_encode"):
  sent=torch.as_tensor(enc.encode(g["node_texts"],convert_to_numpy=True),dtype=torch.float32)
  toks=[]
  for text in g["node_texts"]:
   ids=enc.tokenizer(text,truncation=False,add_special_tokens=True)["input_ids"]
   truncated += len(ids)>enc.max_seq_length
   toks.append(enc.encode(text,output_value="token_embeddings",convert_to_tensor=True).detach().cpu().float())
  z=dict(g);z.update(x_sentence=sent,x_token_mean=torch.stack([x.mean(0) for x in toks]),x_token_ori=toks);out.append(z)
 Path(cache).parent.mkdir(parents=True,exist_ok=True);torch.save(out,cache);write_json(cp,{"identity":contract,"node_texts_over_official_encoder_limit":truncated,"rows":len(out)})
 return out

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def batch_args(rows,device):
 xs=torch.cat([r["x_sentence"] for r in rows]).to(device);xm=torch.cat([r["x_token_mean"] for r in rows]).to(device)
 toks=[[x.to(device) for x in r["x_token_ori"]] for r in rows];edges=[];bv=[];off=0
 for i,r in enumerate(rows):edges.append(torch.as_tensor(r["edge_index"],dtype=torch.long,device=device)+off);bv.extend([i]*len(r["nodes"]));off+=len(r["nodes"])
 return xs,xm,toks,torch.cat(edges,1),torch.tensor(bv,dtype=torch.long,device=device)
def train(model,fuse,rows,device,epochs,batch_size,alpha,lr,wd,seed,out):
 clean=[r for r in rows if r["gold_verdict"]=="clean_safe"]
 groups=collections.defaultdict(list)
 for r in clean:groups[len(r["nodes"])].append(r)
 opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd);hist=[]
 for ep in range(1,epochs+1):
  model.train();losses=[];rng=random.Random(seed+ep)
  for n,group in sorted(groups.items()):
   rng.shuffle(group)
   for start in tqdm(range(0,len(group),batch_size),desc=f"official_xgguard_epoch_{ep}_nodes{n}"):
    b=group[start:start+batch_size]
    if len(b)<2:continue
    xs,xm,toks,edge,bv=batch_args(b,device);opt.zero_grad(set_to_none=True)
    es,et,cs,ct=model.forward(xs,xm,toks,edge,bv);perm=torch.randperm(cs.size(0),device=device)
    ss=model.inference(es,cs,bv);sn=model.inference(es,cs[perm],bv);ts,_=model.inference_token(et,ct,bv);tn,_=model.inference_token(et,ct[perm],bv)
    pos=fuse(ts,ss).reshape(-1)/2;neg=fuse(tn,sn).reshape(-1)/2
    loss=F.binary_cross_entropy_with_logits(pos,torch.zeros_like(pos))+alpha*F.binary_cross_entropy_with_logits(neg,torch.ones_like(neg));loss.backward();opt.step();losses.append(float(loss.detach()))
  hist.append({"epoch":ep,"loss":float(np.mean(losses)),"batches":len(losses)});print(json.dumps(hist[-1]))
 torch.save(model.state_dict(),out);return hist,len(clean)
def score(model,fuse,rows,device):
 model.eval();out=[]
 with torch.no_grad():
  for r in tqdm(rows,desc="official_xgguard_score"):
   xs=r["x_sentence"].to(device);xm=r["x_token_mean"].to(device);t=[x.to(device) for x in r["x_token_ori"]];e=torch.as_tensor(r["edge_index"],dtype=torch.long,device=device)
   es,et,cs,ct=model.forward(xs,xm,t,e);ss=model.inference(es,cs);ts,detail=model.inference_token(et,ct);v=fuse(ss,ts).detach().cpu().numpy()
   z=dict(r);z["node_scores"]={n:float(x) for n,x in zip(r["nodes"],v)};out.append(z)
 return out
def native(rows):
 y=[];s=[]
 for r in rows:
  y.extend([int(n in r["gold_agents"]) for n in r["nodes"]]);s.extend(r["node_scores"].values())
 return {"nodes":len(y),"positive_nodes":sum(y),"node_auroc":roc_auc_score(y,s) if len(set(y))==2 else None,"node_auprc":average_precision_score(y,s) if sum(y) else None}
def candidate_scores(r):
 ns=r["node_scores"];vals={}
 for c in r["candidates"]:
  cid=str(c["id"]);typ=str(c.get("type","")).lower()
  if typ=="node":v=ns.get(str(c.get("agent")),-math.inf)
  elif typ=="edge":v=max(ns.get(str(c.get("source")),-math.inf),ns.get(str(c.get("target")),-math.inf))
  elif typ=="tool":v=ns.get(str(c.get("agent")),-math.inf)
  else:v=max(ns.values())
  vals[cid]=float(v)
 return vals
def pred_scope(ids,cmap):
 ts={str(cmap[i].get("type","unknown")) for i in ids if i in cmap}
 return "none" if not ts else (next(iter(ts)) if len(ts)==1 else "multi")
def records(rows,lo,hi,ct):
 out=[]
 for r in rows:
  gs=max(r["node_scores"].values());pred=VERDICTS[0] if gs<lo else (VERDICTS[1] if gs<hi else VERDICTS[2]);cs=candidate_scores(r);ids=[i for i,v in cs.items() if v>=ct];cm={str(c["id"]):c for c in r["candidates"]}
  out.append({"run_id":r["run_id"],"gold":r["gold_verdict"],"pred":pred,"gold_scope":r["gold_scope"],"pred_scope":pred_scope(ids,cm),"gold_components":r["gold_components"],"pred_components":ids,"node_scores":r["node_scores"]})
 return out
def metrics(rs):
 gold=[r["gold"] for r in rs];pred=[r["pred"] for r in rs];rep=classification_report(gold,pred,labels=list(VERDICTS),zero_division=0,output_dict=True)
 gb=[x=="attack_success" for x in gold];pb=[x=="attack_success" for x in pred];tp=fp=fn=hit=exact=scope=n=0
 for r in rs:
  if r["gold"]!="attack_success" or not r["gold_components"]:continue
  g,p=set(r["gold_components"]),set(r["pred_components"]);tp+=len(g&p);fp+=len(p-g);fn+=len(g-p);hit+=bool(g&p);exact+=g==p;scope+=r["gold_scope"]==r["pred_scope"];n+=1
 prec=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0
 return {"n":len(rs),"three_class_accuracy":accuracy_score(gold,pred),"three_class_report":rep,"binary_accuracy":accuracy_score(gb,pb),"localization":{"n":n,"component_micro_precision":prec,"component_micro_recall":rec,"component_micro_f1":2*prec*rec/(prec+rec) if prec+rec else 0,"component_hit_rate":hit/n if n else 0,"component_exact_match":exact/n if n else 0,"scope_accuracy":scope/n if n else 0}}
def calibrate(rows):
 scores=np.asarray([max(r["node_scores"].values()) for r in rows]);qs=np.unique(np.quantile(scores,np.linspace(.02,.98,49)));best=None
 for lo in qs:
  for hi in qs[qs>lo]:
   rs=records(rows,float(lo),float(hi),math.inf);m=metrics(rs);v=m["three_class_report"]["macro avg"]["f1-score"]
   if best is None or v>best[0]:best=(v,float(lo),float(hi))
 cvals=np.asarray([v for r in rows for v in candidate_scores(r).values()]);bestc=None
 for ct in np.unique(np.quantile(cvals,np.linspace(.05,.95,37))):
  m=metrics(records(rows,best[1],best[2],float(ct)));v=m["localization"]["component_micro_f1"]
  if bestc is None or v>bestc[0]:bestc=(v,float(ct))
 return {"verdict_lower":best[1],"verdict_upper":best[2],"component_threshold":bestc[1],"selection":"validation labels only; maximize 3-way macro-F1 then localization micro-F1"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--official-dir",type=Path,required=True);p.add_argument("--train",type=Path,required=True);p.add_argument("--validation",type=Path,required=True);p.add_argument("--test",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--cache-dir",type=Path,required=True);p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--seed",type=int,default=3701);p.add_argument("--smoke-only",action="store_true");a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
 Model,fuse,identity=official_model(a.official_dir);sources={k:sha(v) for k,v in (("train",a.train),("validation",a.validation),("test",a.test))};raw={k:[convert(x) for x in read(v)] for k,v in (("train",a.train),("validation",a.validation),("test",a.test))}
 if a.smoke_only:raw={k:v[:100] for k,v in raw.items()}
 for k,v in raw.items():
  # A one-agent run is a valid degenerate graph. ``convert`` adds its self-loop,
  # so the official GCN can process it without inventing or dropping a node.
  # Contrastive training still uses at least two same-size graphs per batch.
  if not v or any(len(x["nodes"])<1 for x in v):raise RuntimeError(f"invalid {k} graphs")
 graph_statistics={k:graph_stats(v) for k,v in raw.items()}
 enc={k:encode(v,a.cache_dir/f"{k}{'_smoke' if a.smoke_only else ''}.pt",sources[k]) for k,v in raw.items()};device=torch.device("cuda");seed_all(a.seed);model=Model(enc["train"][0]["x_sentence"].shape[1]).to(device)
 a.output_dir.mkdir(parents=True,exist_ok=True);ck=a.output_dir/"official_oursmethod.pt";hist,nclean=train(model,fuse,enc["train"],device,1 if a.smoke_only else a.epochs,a.batch_size,1e-4,1e-4,2e-4,a.seed,ck)
 val=score(model,fuse,enc["validation"],device);cal=calibrate(val);test=score(model,fuse,enc["test"],device);pred=records(test,cal["verdict_lower"],cal["verdict_upper"],cal["component_threshold"]);m=metrics(pred);m.update({"method":"Official XG-Guard OursMethod + V22 adapter","official_identity":identity,"native_validation":native(val),"native_test":native(test),"calibration":cal,"training_clean_rows":nclean,"graph_statistics":graph_statistics,"single_node_policy":"retain as a valid self-loop graph; no synthetic nodes and no sample filtering","source_sha256":sources,"smoke_only":a.smoke_only,"adaptation":"agent-node anomaly is native; V22 verdict/component projection is validation-calibrated"});write_json(a.output_dir/"metrics.json",m);write_jsonl(a.output_dir/"predictions.jsonl",pred);write_json(a.output_dir/"RUN_COMPLETE.json",{"status":"PASS","metrics_sha256":sha(a.output_dir/"metrics.json"),"official_identity":identity,"smoke_only":a.smoke_only});print(json.dumps(m,indent=2))
if __name__=="__main__":main()
