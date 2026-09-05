#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,py_compile,subprocess
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
 pkg=a.repo/'SFT/auditor_agent_v23_final_experiments';core=a.repo/'SFT/auditor_agent_sft_v23_final_package';v19=a.repo/'SFT/auditor_agent_sft_v19_qualityfix_package';v22=a.repo/'SFT/auditor_agent_sft_v22_all_package'
 required=[pkg/'scripts/run_v23_experiment_suite.py',pkg/'scripts/build_heldout_splits.py',pkg/'scripts/evaluate_component_policies.py',pkg/'scripts/render_experiment_tables.py',pkg/'scripts/v22_xgguard_official_adapter.py',a.repo/'SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_unsupervised_graph_baselines.py',a.repo/'SFT/auditor_agent_gnn_baselines_package/server_scripts/v19_component_gnn_multitask.py',v19/'server_scripts/train_qwen3_lora_sft_v19.py',v19/'server_scripts/eval_qwen3_fullschema_v19.py',v19/'server_scripts/modernbert_multitask_v19.py',v22/'scripts/score_predictions_by_track.py',v22/'scripts/v22_plain_hetero_agent.py',v22/'scripts/filter_v22_expanded_modernbert_context.py',core/'scripts/audit_v23_qwen_sft_contract.py']
 shells=[a.repo/'run_v23_h100.sh',a.repo/'run_v23_aux_h100.sh',pkg/'server_scripts/run_v23_all_experiments.sh',pkg/'server_scripts/run_v23_heldout_suite.sh',pkg/'server_scripts/run_v23_component_policies_once.sh',pkg/'server_scripts/run_v22_official_gsafe_tam_once.sh',pkg/'server_scripts/run_v22_official_xgguard_once.sh',pkg/'server_scripts/run_v23_official_blindguard_once.sh',core/'server_scripts/run_v23_plain_qwen_sft_once.sh',core/'server_scripts/run_v23_internlm3_sft_once.sh',core/'server_scripts/run_v23_modernbert_once.sh',core/'server_scripts/run_v23_bounded_agent_once.sh']
 missing=[str(f) for f in required+shells if not f.is_file()]
 if missing:raise RuntimeError(f'missing runtime files: {missing}')
 for f in required:py_compile.compile(str(f),doraise=True)
 for f in shells:subprocess.run(['bash','-n',str(f)],check=True)
 out={'status':'PASS','version':'V23-preflight-v3','data':str(a.data.resolve()),'checks':checks,'python_files_compiled':len(required),'shell_entrypoints_checked':len(shells)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
