#!/usr/bin/env python3
"""Dependency-aware V23 runner: safe on one GPU, parallel on multiple GPUs."""
from __future__ import annotations
import argparse,json,os,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--experiments',type=Path,required=True);a=p.parse_args()
 env=os.environ.copy();env.update(REPO=str(a.repo),V23_DATA_DIR=str(a.data),V23_RUN=str(a.run),V23_EXPERIMENT_RUN=str(a.experiments));a.experiments.mkdir(parents=True,exist_ok=True);(a.experiments/'logs').mkdir(exist_ok=True)
 core=a.repo/'SFT/auditor_agent_sft_v23_final_package';pkg=a.repo/'SFT/auditor_agent_v23_final_experiments'
 def task(key,name,cmd,marker,deps=(),gpu_gb=0,extra=None,always=False):return {'key':key,'name':name,'cmd':cmd,'marker':marker,'deps':set(deps),'gpu_gb':gpu_gb,'extra':extra or {},'always':always}
 tasks=[
  task('preflight','Preflight contracts',['python',str(pkg/'scripts/preflight_v23_experiments.py'),'--repo',str(a.repo),'--data',str(a.data),'--output',str(a.experiments/'PREFLIGHT.json')],a.experiments/'PREFLIGHT.json',always=True),
  task('qwen','Qwen3-8B SFT + validation + test',['bash',str(core/'server_scripts/run_v23_plain_qwen_sft_once.sh')],a.run/'V23_SFT_COMPLETE.json',['preflight'],34),
  task('intern','InternLM3-8B SFT + validation + test',['bash',str(core/'server_scripts/run_v23_internlm3_sft_once.sh')],a.run/'INTERNLM3_SFT_COMPLETE.json',['preflight'],34),
  task('modern','ModernBERT train + common evaluation',['bash',str(core/'server_scripts/run_v23_modernbert_once.sh')],a.run/'MODERNBERT_COMPLETE.json',['preflight'],12),
  task('heldout','V23 held-out ModernBERT folds',['bash',str(pkg/'server_scripts/run_v23_heldout_suite.sh')],a.experiments/'SUPPLEMENT_SUITE_COMPLETE.json',['preflight'],12),
  task('gsafe','G-Safeguard V23',['bash',str(pkg/'server_scripts/run_v22_official_gsafe_tam_once.sh')],a.experiments/'baselines/gsafeguard_official_v23_v1/test/metrics.json',['preflight'],10,{'GNN_METHODS':'gat'}),
  task('tam','TAM V23',['bash',str(pkg/'server_scripts/run_v22_official_gsafe_tam_once.sh')],a.experiments/'baselines/tam_official_v23_v1/test/metrics.json',['gsafe'],10,{'GNN_METHODS':'tam'}),
  task('xg','XG-Guard V23',['bash',str(pkg/'server_scripts/run_v22_official_xgguard_once.sh')],a.experiments/'baselines/xgguard_official_v23_v1/full/metrics.json',['preflight'],10),
  task('blind','BlindGuard SCL V23',['bash',str(pkg/'server_scripts/run_v23_official_blindguard_once.sh')],a.experiments/'baselines/blindguard_official_v23_v1/test/metrics.json',['preflight'],12),
  task('components','Fixed and rule component policies',['bash',str(pkg/'server_scripts/run_v23_component_policies_once.sh')],a.experiments/'components/COMPONENT_POLICIES_COMPLETE.json',['qwen','modern']),
  task('agent','Learned-router bounded agent',['bash',str(core/'server_scripts/run_v23_bounded_agent_once.sh')],a.run/'bounded_agent_common6167/V23_AGENT_COMPLETE.json',['qwen','modern'],24),
  task('render','Render four publication tables',['python',str(pkg/'scripts/render_experiment_tables.py'),'--run-dir',str(a.run),'--supplement-dir',str(a.experiments),'--output-dir',str(a.experiments/'tables')],a.experiments/'tables/TABLE_STATUS.json',['qwen','intern','modern','heldout','gsafe','tam','xg','blind','components','agent'],always=True)]
 gpu_ids=[x.strip() for x in env.get('V23_GPUS',env.get('GPU','0')).split(',') if x.strip()]
 if not gpu_ids:raise SystemExit('V23_GPUS/GPU resolved to an empty GPU list')
 def detected_gb(gpu):
  override=env.get('V23_GPU_MEMORY_GB')
  if override:return float(override)
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'],text=True)
   values={x.split(',')[0].strip():float(x.split(',')[1].strip())/1024 for x in raw.splitlines()};return values[gpu]
  except Exception:return 24.0
 utilization=float(env.get('V23_GPU_MEMORY_UTILIZATION','0.85'));capacities={g:detected_gb(g)*utilization for g in gpu_ids};reserved={g:0.0 for g in gpu_ids}
 print(f"SCHEDULER: usable_gpu_gb={capacities}; same-GPU jobs run concurrently only inside the reservation budget",flush=True)
 lock=threading.Lock();start=time.time();results=[]
 def run(t,gpu):
  if t['marker'].is_file() and not t['always']:return {'task':t['name'],'key':t['key'],'status':'SKIPPED_COMPLETE','gpu':gpu,'seconds':0}
  e=env.copy();e.update(t['extra']);
  if gpu is not None:e['GPU']=gpu
  log=a.experiments/'logs'/f"{t['key']}.log";beg=time.time();print(f"START {t['key']} gpu={gpu if gpu is not None else 'CPU'}",flush=True)
  try:
   with log.open('a',encoding='utf-8') as f:
    x=subprocess.Popen(t['cmd'],cwd=a.repo,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in x.stdout or ():
     f.write(line);f.flush()
     with lock:print(f"[{t['key']}] {line}",end='',flush=True)
    rc=x.wait()
  finally:
   pass
  return {'task':t['name'],'key':t['key'],'status':'PASS' if rc==0 else 'FAIL','returncode':rc,'gpu':gpu,'seconds':time.time()-beg,'log':str(log)}
 task_by_key={t['key']:t for t in tasks};pending=dict(task_by_key);complete=set();running={}
 with ThreadPoolExecutor(max_workers=16) as ex:
  while pending or running:
   for key,t in list(pending.items()):
    if not t['deps']<=complete:continue
    gpu=None
    if t['gpu_gb']:
     gpu=next((g for g in gpu_ids if reserved[g]==0 or reserved[g]+t['gpu_gb']<=capacities[g]),None)
     if gpu is None:continue
     reserved[gpu]+=t['gpu_gb']
    running[ex.submit(run,t,gpu)]=key;del pending[key]
   if not running:raise SystemExit(f"scheduler dependency deadlock: {sorted(pending)}")
   finished,_=wait(running,return_when=FIRST_COMPLETED)
   for f in finished:
    key=running.pop(f);r=f.result();t=task_by_key[key]
    if t['gpu_gb']:reserved[r['gpu']]-=t['gpu_gb']
    results.append(r);(a.experiments/'PROGRESS.json').write_text(json.dumps({'gpus':gpu_ids,'usable_gpu_gb':capacities,'reserved_gpu_gb':reserved,'done':results,'pending':sorted(pending)},indent=2),encoding='utf-8')
    if r['status']=='FAIL':raise SystemExit(f"{r['task']} failed; see {r['log']}")
    complete.add(key)
 status=json.loads((a.experiments/'tables/TABLE_STATUS.json').read_text());out={'version':'V23-final-four-table-suite-v3','pipeline_status':'PASS','scheduler':'dependency-and-memory-aware','gpus':gpu_ids,'usable_gpu_gb':capacities,'table_status':status['status'],'required_tbd_cells':status['required_tbd_cells'],'seconds':time.time()-start,'tasks':results}
 if status['required_tbd_cells']:raise SystemExit('Required V23 table cells remain TBD; completion marker withheld')
 (a.experiments/'V23_EXPERIMENTS_COMPLETE.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
