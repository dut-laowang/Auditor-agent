from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "password_key": re.compile(r"[\"']?password[\"']?\s*[:=]", re.I),
    "phone_7plus": re.compile(r"(?<!\d)\d{7,15}(?!\d)"),
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v22',type=Path,required=True);ap.add_argument('--v23',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    old=json.loads((a.v22/'ASSEMBLY_MANIFEST.json').read_text(encoding='utf-8'));counts=Counter();examples={k:[] for k in PATTERNS}
    for track in old['tracks']:
        for split in ('train','validation','test'):
            old_n=old['tracks'][track][split]['rows']
            with (a.v23/track/f'{split}.jsonl').open(encoding='utf-8') as f:
                for pos,line in enumerate((x for x in f if x.strip()),1):
                    row=json.loads(line);origin='V22' if pos<=old_n else 'V23_increment';visible=' '.join(x['content'] for x in row['messages'][:2])
                    for name,pattern in PATTERNS.items():
                        if pattern.search(visible):
                            counts[(origin,name)]+=1
                            if len(examples[name])<10:examples[name].append({'origin':origin,'run_id':row['metadata']['run_id'],'track':track,'split':split})
    report={'version':'V23-visible-sensitive-shape-audit-v1','interpretation':'Pattern presence only. AppWorld uses synthetic identities and credentials; these are observable trajectory contents, not gold-label fields.','row_hits':{'/'.join(k):v for k,v in sorted(counts.items())},'examples':examples}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'row_hits':report['row_hits']},indent=2))
if __name__=='__main__':main()
