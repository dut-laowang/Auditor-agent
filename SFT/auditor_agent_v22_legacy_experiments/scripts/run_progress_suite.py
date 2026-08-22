#!/usr/bin/env python3
"""Resumable task runner with global progress, elapsed time, ETA, and live logs."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

def clock(x):
    x=max(0,int(x)); h,x=divmod(x,3600); m,s=divmod(x,60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    pkg=a.repo/"SFT/auditor_agent_v22_legacy_experiments"; a.out.mkdir(parents=True,exist_ok=True)
    py=os.environ.get("PYTHON","python"); env=os.environ.copy()
    env.update(V22_LEGACY_RUN=str(a.run_dir),V22_SUPPLEMENT_RUN=str(a.out),REPO=str(a.repo),METHODS="modernbert",RUN_XGGUARD_TEST=env.get("RUN_XGGUARD_TEST","1"))
    tasks=[
      ("Protocol/data preflight + 3 ModernBERT held-out folds",["bash",str(pkg/"server_scripts/run_v22_legacy_supplement_suite.sh")],7200,True),
      ("Qwen3-8B/32B base full-test inference (no training)",["bash",str(pkg/"server_scripts/run_v22_base_llms_once.sh")],28800,env.get("RUN_BASE_LLMS","1")=="1"),
      ("Public XG-Guard validation/test",["bash",str(pkg/"server_scripts/run_v22_legacy_external_baselines.sh")],5400,env.get("RUN_EXTERNAL_BASELINES","1")=="1"),
      ("Bounded Agent full-test evaluation",["bash",str(a.repo/"SFT/auditor_agent_sft_v22_all_package/server_scripts/run_v22_plain_hetero_agent_full_test_once.sh")],3600,env.get("RUN_AGENT_FULL","1")=="1"),
      ("Render and validate five tables",[py,str(pkg/"scripts/render_experiment_tables.py"),"--run-dir",str(a.run_dir),"--supplement-dir",str(a.out),"--output-dir",str(a.out/"tables")],30,True)]
    tasks=[x[:3] for x in tasks if x[3]]; start=time.time(); done=[]
    for i,(name,cmd,estimate) in enumerate(tasks,1):
        elapsed=time.time()-start; left=sum(x[2] for x in tasks[i-1:]); filled=int(20*(i-1)/len(tasks)); bar="#"*filled+"-"*(20-filled)
        print(f"\n[{bar}] task {i}/{len(tasks)} START: {name}\nelapsed={clock(elapsed)} estimated_remaining={clock(left)}",flush=True)
        t=time.time(); log=a.out/f"task_{i:02d}.log"
        if name.startswith("Protocol/data preflight") and (a.out/"SUPPLEMENT_SUITE_COMPLETE.json").is_file():
            duration=time.time()-t; done.append({"task":i,"name":name,"returncode":0,"seconds":duration,"status":"SKIPPED_COMPLETE","log":str(log)})
            print(f"task {i}/{len(tasks)} SKIP: validated completion marker exists",flush=True)
            continue
        with log.open("a",encoding="utf-8") as f:
            p=subprocess.Popen(cmd,cwd=a.repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            assert p.stdout is not None
            for line in p.stdout:
                f.write(line); f.flush(); print(f"[task {i}/{len(tasks)}] {line}",end="",flush=True)
            p.wait()
        duration=time.time()-t; done.append({"task":i,"name":name,"returncode":p.returncode,"seconds":duration,"log":str(log)})
        (a.out/"PROGRESS.json").write_text(json.dumps({"completed":i,"total":len(tasks),"elapsed_seconds":time.time()-start,"tasks":done},indent=2),encoding="utf-8")
        if p.returncode:
            print(log.read_text(encoding="utf-8",errors="replace")[-5000:]); raise SystemExit(f"task {i} failed; see {log}")
        print(f"task {i}/{len(tasks)} DONE in {clock(duration)}; log={log}",flush=True)
    print(f"\n[{'#'*20}] ALL {len(tasks)} TASKS DONE; total={clock(time.time()-start)}\noutput={a.out}")
if __name__=="__main__": main()
