from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from tqdm.auto import tqdm

LEAK = __import__('re').compile(r"ACI_[A-Z0-9_]+|\baci_[a-z0-9_]+\b|\bEND_NEGOTIATION\b|success_marker|success_markers|attack_metadata|attack_id|marker_check|\[Injected[^\]]*\]|offline verifier|attack-success index|labeled as attack-success", __import__('re').I)


def render(tok, messages, generation):
    try: return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-8B');ap.add_argument('--revision',required=True);ap.add_argument('--max-len',type=int,default=12288);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--splits',nargs='+',choices=('train','validation','test'),default=('train','validation','test'));a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=True)
    report={'version':'V23-Qwen-SFT-contract-audit-v1','model':a.model,'revision':a.revision,'max_len':a.max_len,'splits':{}}
    total_errors=0
    for split in a.splits:
        rows=[json.loads(x) for x in (a.data_dir/f'{split}.jsonl').open(encoding='utf-8') if x.strip()]
        issues=Counter();examples=[];max_full=max_prompt=max_target=0
        batches=range(0,len(rows),a.batch_size)
        progress=tqdm(
            batches,
            total=(len(rows)+a.batch_size-1)//a.batch_size,
            desc=f'{a.model} audit {split}',
            unit='batch',
            dynamic_ncols=True,
        )
        for start in progress:
            batch=rows[start:start+a.batch_size]
            full_text=[render(tok,r['messages'],False) for r in batch]
            prompt_text=[render(tok,r['messages'][:2],True) for r in batch]
            full_len=tok(full_text,add_special_tokens=False,return_length=True)['length']
            prompt_len=tok(prompt_text,add_special_tokens=False,return_length=True)['length']
            for off,(r,fl,pl) in enumerate(zip(batch,full_len,prompt_len)):
                max_full=max(max_full,int(fl));max_prompt=max(max_prompt,int(pl));max_target=max(max_target,int(fl)-int(pl))
                row_issues=[]
                if fl>a.max_len: row_issues.append('full_overflow')
                if pl>=a.max_len: row_issues.append('prompt_overflow')
                if fl<=pl: row_issues.append('no_assistant_supervision')
                if LEAK.search(json.dumps(r['messages'][:2],ensure_ascii=False)): row_issues.append('visible_leak_pattern')
                for issue in row_issues: issues[issue]+=1
                if row_issues and len(examples)<30: examples.append({'position':start+off,'run_id':r['metadata']['run_id'],'issues':row_issues,'full_tokens':int(fl),'prompt_tokens':int(pl)})
        total_errors+=sum(issues.values());report['splits'][split]={'rows':len(rows),'issues':dict(issues),'examples':examples,'max_full_tokens':max_full,'max_prompt_tokens':max_prompt,'max_supervised_tokens':max_target}
    report['status']='PASS' if not total_errors else 'FAIL';report['total_issues']=total_errors;a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
    if total_errors:raise SystemExit(1)
if __name__=='__main__':main()
