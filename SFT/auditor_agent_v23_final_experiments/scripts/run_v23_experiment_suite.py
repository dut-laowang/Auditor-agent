#!/usr/bin/env python3
"""Dependency-aware V23 runner: safe on one GPU, parallel on multiple GPUs."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path

EXPECTED_ROWS={'train':30619,'validation':7018,'test':6207}

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_json(path):
 try:return json.loads(path.read_text(encoding='utf-8'))
 except (OSError,json.JSONDecodeError,TypeError):return None

def valid_metric(path,rows,data_sha=None):
 value=load_json(path)
 if not isinstance(value,dict) or value.get('n')!=rows:return False
 required=('three_class_accuracy','binary_accuracy','three_class_report','localization')
 if any(k not in value for k in required):return False
 if data_sha is not None and value.get('data_sha256')!=data_sha:return False
 return True

def valid_status(path,rows=None):
 value=load_json(path)
 return isinstance(value,dict) and value.get('status')=='PASS' and (rows is None or value.get('rows')==rows)

def valid_lm_eval(directory,rows,data_sha):
 metric=directory/'metrics.json';contract=load_json(directory/'EVAL_CONTRACT.json')
 return valid_metric(metric,rows) and isinstance(contract,dict) and contract.get('rows')==rows and contract.get('data_sha256')==data_sha

def valid_heldout(root):
 expected={'topology__tree':765,'surface__message':3041,'scenario__research':1064}
 return valid_status(root/'SUPPLEMENT_SUITE_COMPLETE.json') and all(valid_metric(root/f'heldout/{fold}/modernbert_ztr_v2/metrics.json',rows) for fold,rows in expected.items())

def valid_components(root):
 return valid_status(root/'COMPONENT_POLICIES_COMPLETE.json',6167) and all(valid_metric(root/f'{name}/metrics.json',6167) for name in ('fixed_cascade','rule_router'))

def valid_agent(root):
 marker=load_json(root/'V23_AGENT_COMPLETE.json');comparison=load_json(root/'AGENT_TEST_COMPARISON.json')
 return isinstance(marker,dict) and marker.get('status')=='PASS' and marker.get('rows')==6167 and isinstance(comparison,dict) and comparison.get('rows')==6167

def choose_gpu(gpu_ids,reserved,capacities,need,exclusive=False,exclusive_running=None):
 exclusive_running=exclusive_running or {g:False for g in gpu_ids}
 candidates=[g for g in gpu_ids if not exclusive_running[g] and reserved[g]+need<=capacities[g] and (not exclusive or reserved[g]==0)]
 return min(candidates,key=lambda g:(reserved[g],gpu_ids.index(g))) if candidates else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--experiments',type=Path,required=True);a=p.parse_args()
 env=os.environ.copy();env.update(REPO=str(a.repo),V23_DATA_DIR=str(a.data),V23_RUN=str(a.run),V23_EXPERIMENT_RUN=str(a.experiments));a.experiments.mkdir(parents=True,exist_ok=True);(a.experiments/'logs').mkdir(exist_ok=True)
 core=a.repo/'SFT/auditor_agent_sft_v23_final_package';pkg=a.repo/'SFT/auditor_agent_v23_final_experiments'
 data_sha={s:sha256(a.data/f'{s}.jsonl') for s in EXPECTED_ROWS}
 git_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
 contract_keys=('QWEN_EPOCHS','QWEN_LR','QWEN_TRAIN_BATCH','QWEN_GRAD_ACCUM','QWEN_EVAL_BATCH','INTERNLM_REVISION','INTERNLM_EPOCHS','INTERNLM_LR','INTERNLM_TRAIN_BATCH','INTERNLM_GRAD_ACCUM','INTERNLM_EVAL_BATCH','MODERN_EPOCHS','MODERN_LR','MODERN_TRAIN_BATCH','MODERN_GRAD_ACCUM','MODERN_EVAL_BATCH','GNN_OFFICIAL_EPOCHS','GNN_GRAD_ACCUM','BLINDGUARD_EPOCHS','XGGUARD_OFFICIAL_EPOCHS','XGGUARD_OFFICIAL_BATCH','XGGUARD_OFFICIAL_LR','XGGUARD_OFFICIAL_WEIGHT_DECAY','XGGUARD_OFFICIAL_ALPHA','AGENT_MAX_VERIFY_RATE','RULE_VERIFY_RATE')
 run_contract={'version':'V23-suite-run-contract-v1','git_commit':git_commit,'data_sha256':data_sha,'configuration':{k:env.get(k) for k in contract_keys}}
 contract_path=a.experiments/'RUN_CONTRACT.json';existing=load_json(contract_path)
 if existing is not None and existing!=run_contract:
  comparable_old={k:v for k,v in existing.items() if k!='git_commit'};comparable_new={k:v for k,v in run_contract.items() if k!='git_commit'}
  if env.get('V23_ALLOW_CODE_UPDATE_RESUME')=='1' and comparable_old==comparable_new and not (a.experiments/'V23_EXPERIMENTS_COMPLETE.json').exists():
   migration=a.experiments/'RUN_CONTRACT_MIGRATIONS.jsonl'
   with migration.open('a',encoding='utf-8') as f:f.write(json.dumps({'from':existing.get('git_commit'),'to':git_commit,'reason':'explicit V23_ALLOW_CODE_UPDATE_RESUME=1'})+'\n')
   contract_path.write_text(json.dumps(run_contract,indent=2),encoding='utf-8');existing=run_contract
  else:raise SystemExit(f'Existing result directory has a different code/data/configuration contract: {contract_path}')
 if existing is None:contract_path.write_text(json.dumps(run_contract,indent=2),encoding='utf-8')
 def task(key,name,cmd,marker,deps=(),gpu_gb=0,extra=None,always=False,validate=None,exclusive=False):return {'key':key,'name':name,'cmd':cmd,'marker':marker,'deps':set(deps),'gpu_gb':gpu_gb,'extra':extra or {},'always':always,'validate':validate,'exclusive':exclusive}
 tasks=[
  task('preflight','Preflight contracts',['python',str(pkg/'scripts/preflight_v23_experiments.py'),'--repo',str(a.repo),'--data',str(a.data),'--output',str(a.experiments/'PREFLIGHT.json')],a.experiments/'PREFLIGHT.json',always=True),
  task('qwen','Qwen3-8B SFT + validation + test',['bash',str(core/'server_scripts/run_v23_plain_qwen_sft_once.sh')],a.run/'V23_SFT_COMPLETE.json',['preflight'],34,validate=lambda:valid_lm_eval(a.run/'qwen3_8b_plain_sft_test',6207,data_sha['test']),exclusive=True),
  task('intern','InternLM3-8B SFT + validation + test',['bash',str(core/'server_scripts/run_v23_internlm3_sft_once.sh')],a.run/'INTERNLM3_SFT_COMPLETE.json',['preflight'],34,validate=lambda:valid_lm_eval(a.run/'internlm3_8b_sft_test',6207,data_sha['test']),exclusive=True),
  task('modern','ModernBERT train + common evaluation',['bash',str(core/'server_scripts/run_v23_modernbert_once.sh')],a.run/'MODERNBERT_COMPLETE.json',['preflight'],12,validate=lambda:valid_status(a.run/'MODERNBERT_COMPLETE.json',6167) and load_json(a.run/'MODERNBERT_COMPLETE.json').get('source_test_sha256')==data_sha['test'] and valid_metric(a.run/'modernbert_test/metrics.json',6167)),
  task('heldout','V23 held-out ModernBERT folds',['bash',str(pkg/'server_scripts/run_v23_heldout_suite.sh')],a.experiments/'SUPPLEMENT_SUITE_COMPLETE.json',['preflight'],12,validate=lambda:valid_heldout(a.experiments)),
  task('gsafe','G-Safeguard V23',['bash',str(pkg/'server_scripts/run_v22_official_gsafe_tam_once.sh')],a.experiments/'baselines/gsafeguard_official_v23_v1/test/metrics.json',['preflight'],10,{'GNN_METHODS':'gat'},validate=lambda:valid_metric(a.experiments/'baselines/gsafeguard_official_v23_v1/test/metrics.json',6207,data_sha['test'])),
  task('tam','TAM V23',['bash',str(pkg/'server_scripts/run_v22_official_gsafe_tam_once.sh')],a.experiments/'baselines/tam_official_v23_v1/test/metrics.json',['gsafe'],10,{'GNN_METHODS':'tam'},validate=lambda:valid_metric(a.experiments/'baselines/tam_official_v23_v1/test/metrics.json',6207,data_sha['test'])),
  task('xg','XG-Guard V23',['bash',str(pkg/'server_scripts/run_v22_official_xgguard_once.sh')],a.experiments/'baselines/xgguard_official_v23_v2/test/metrics.json',['preflight'],12,validate=lambda:valid_metric(a.experiments/'baselines/xgguard_official_v23_v2/test/metrics.json',6207,data_sha['test'])),
  task('blind','BlindGuard SCL V23',['bash',str(pkg/'server_scripts/run_v23_official_blindguard_once.sh')],a.experiments/'baselines/blindguard_official_v23_v1/test/metrics.json',['preflight'],12,validate=lambda:valid_metric(a.experiments/'baselines/blindguard_official_v23_v1/test/metrics.json',6207,data_sha['test'])),
  task('components','Fixed and rule component policies',['bash',str(pkg/'server_scripts/run_v23_component_policies_once.sh')],a.experiments/'components/COMPONENT_POLICIES_COMPLETE.json',['qwen','modern'],validate=lambda:valid_components(a.experiments/'components')),
  task('agent','Learned-router bounded agent',['bash',str(core/'server_scripts/run_v23_bounded_agent_once.sh')],a.run/'bounded_agent_common6167/V23_AGENT_COMPLETE.json',['qwen','modern'],24,validate=lambda:valid_agent(a.run/'bounded_agent_common6167')),
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
 utilization=float(env.get('V23_GPU_MEMORY_UTILIZATION','0.85'));capacities={g:detected_gb(g)*utilization for g in gpu_ids};reserved={g:0.0 for g in gpu_ids};exclusive_running={g:False for g in gpu_ids}
 print(f"SCHEDULER: usable_gpu_gb={capacities}; same-GPU jobs run concurrently only inside the reservation budget",flush=True)
 lock=threading.Lock();start=time.time();results=[];processes={};stop=threading.Event()
 def run(t,gpu):
  if t['marker'].is_file() and not t['always'] and (t['validate'] is None or t['validate']()):return {'task':t['name'],'key':t['key'],'status':'SKIPPED_VALIDATED','gpu':gpu,'seconds':0}
  e=env.copy();e.update(t['extra']);
  if gpu is not None:e['GPU']=gpu
  log=a.experiments/'logs'/f"{t['key']}.log";beg=time.time();print(f"START {t['key']} gpu={gpu if gpu is not None else 'CPU'}",flush=True)
  rc=127
  try:
   with log.open('a',encoding='utf-8') as f:
    x=subprocess.Popen(t['cmd'],cwd=a.repo,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    with lock:processes[t['key']]=x
    for line in x.stdout or ():
     f.write(line);f.flush()
     with lock:print(f"[{t['key']}] {line}",end='',flush=True)
    rc=x.wait()
    with lock:processes.pop(t['key'],None)
  except Exception as exc:
   with log.open('a',encoding='utf-8') as f:f.write(f'\nRUNNER_EXCEPTION: {exc!r}\n')
  return {'task':t['name'],'key':t['key'],'status':'PASS' if rc==0 else 'FAIL','returncode':rc,'gpu':gpu,'seconds':time.time()-beg,'log':str(log)}
 task_by_key={t['key']:t for t in tasks};pending=dict(task_by_key);complete=set();running={}
 with ThreadPoolExecutor(max_workers=16) as ex:
  while pending or running:
   for key,t in list(pending.items()):
    if not t['deps']<=complete:continue
    gpu=None
    if t['gpu_gb']:
     gpu=choose_gpu(gpu_ids,reserved,capacities,t['gpu_gb'],t['exclusive'],exclusive_running)
     if gpu is None:continue
     reserved[gpu]+=t['gpu_gb']
     if t['exclusive']:exclusive_running[gpu]=True
    running[ex.submit(run,t,gpu)]=key;del pending[key]
   if not running:raise SystemExit(f"scheduler dependency deadlock: {sorted(pending)}")
   finished,_=wait(running,return_when=FIRST_COMPLETED)
   for f in finished:
    key=running.pop(f);r=f.result();t=task_by_key[key]
    if t['gpu_gb']:
     reserved[r['gpu']]-=t['gpu_gb']
     if t['exclusive']:exclusive_running[r['gpu']]=False
    results.append(r);(a.experiments/'PROGRESS.json').write_text(json.dumps({'gpus':gpu_ids,'usable_gpu_gb':capacities,'reserved_gpu_gb':reserved,'done':results,'pending':sorted(pending)},indent=2),encoding='utf-8')
    if r['status']=='FAIL':
     stop.set()
     with lock:active=list(processes.values())
     for process in active:
      try:os.killpg(process.pid,signal.SIGTERM)
      except (ProcessLookupError,PermissionError):pass
     raise SystemExit(f"{r['task']} failed; concurrent jobs terminated; see {r['log']}")
    complete.add(key)
 status=json.loads((a.experiments/'tables/TABLE_STATUS.json').read_text());out={'version':'V23-final-four-table-suite-v3','pipeline_status':'PASS','scheduler':'dependency-and-memory-aware','gpus':gpu_ids,'usable_gpu_gb':capacities,'table_status':status['status'],'required_tbd_cells':status['required_tbd_cells'],'seconds':time.time()-start,'tasks':results}
 if status['required_tbd_cells']:raise SystemExit('Required V23 table cells remain TBD; completion marker withheld')
 (a.experiments/'V23_EXPERIMENTS_COMPLETE.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
