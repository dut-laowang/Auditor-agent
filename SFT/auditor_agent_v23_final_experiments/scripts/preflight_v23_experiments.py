#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,py_compile
from pathlib import Path
def count(p):return sum(1 for x in p.open(encoding="utf-8") if x.strip())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads((a.data/'COMBINED_MANIFEST.json').read_text(encoding='utf-8'));exp={'train':30619,'validation':7018,'test':6207};checks=[]
 if m.get('version')!='V23-ALL-expanded-2x2-combined-final-v1':raise RuntimeError('wrong V23 manifest version')
 for s,n in exp.items():
  d=a.data/f'{s}.jsonl';i=a.data/f'{s}_track_index.jsonl';e=m['splits'][s]
  ok=count(d)==n==e['rows'] and count(i)==n and sha(d)==e['sha256'] and sha(i)==e['index_sha256'];checks.append({'split':s,'rows':n,'hash_and_index':ok})
  if not ok:raise RuntimeError(f'{s} contract failed')
 pkg=a.repo/'SFT/auditor_agent_v23_final_experiments';required=[pkg/'scripts/run_v23_experiment_suite.py',pkg/'scripts/evaluate_component_policies.py',pkg/'scripts/render_experiment_tables.py']
 for f in required:py_compile.compile(str(f),doraise=True)
 out={'status':'PASS','version':'V23-preflight-v1','data':str(a.data.resolve()),'checks':checks,'python_files_compiled':len(required)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
