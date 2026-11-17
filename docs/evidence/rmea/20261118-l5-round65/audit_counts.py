"""Recompute the frozen descriptive criterion from published aggregate counts."""
import json
from pathlib import Path
def main():
 p=Path(__file__).resolve().parent
 x=json.loads((p/'original-score-counts.json').read_bytes());rows=x['cells']
 assert len(rows)==30 and len({(r['seed'],r['head'],r['split']) for r in rows})==30
 assert all(r['original_confusion_exact'] and r['total']=={'valid':19584,'test':13418}[r['split']] for r in rows)
 for r in rows:
  n,k=r['eligible'],r['strict_gold_winner'];assert 0<=k<=n
  expected='UNDEFINED_EMPTY' if n==0 else ('PASS_DEFINED' if 2*k>n else 'FAIL_DEFINED')
  assert r['criterion']==expected
  assert r['proportion']==(k/n if n else None)
 criterion='FAIL_DEFINED_CELL' if any(r['criterion']=='FAIL_DEFINED' for r in rows) else ('UNASSESSABLE_EMPTY' if any(r['criterion']=='UNDEFINED_EMPTY' for r in rows) else 'ALL30_DEFINED_PASS')
 assert x['allcell_criterion']==criterion
 for key,s in x['summary'].items():
  head,split=key.split(':');a=[r for r in rows if r['head']==head and r['split']==split]
  n=sum(r['eligible'] for r in a);k=sum(r['strict_gold_winner'] for r in a)
  assert s['eligible']==n and s['strict_gold_winner']==k and s['pooled_ratio']==k/n
 assert x['candidate']=='NONE' and x['L5']=='NOT_ACCEPTED' and not x['model_training']
 print(json.dumps(dict(status='PASS_PUBLIC_COUNT_RECOMPUTATION',cells=30,criterion=criterion,L5='NOT_ACCEPTED')))
if __name__=='__main__':main()
