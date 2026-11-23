"""Collect compact evidence and independently recompute fixed replay outputs."""
from pathlib import Path
import json
import sys

r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code=r'''
from pathlib import Path
from collections import defaultdict
import hashlib,json,os
import numpy as np
r=Path('/hy-tmp/Q-Attention/control/historical-type-replay-20261123');o=r/'output'
summary=json.loads((o/'summary.json').read_bytes());design=json.loads((r/'design.json').read_bytes());execution=json.loads((r/'execution.json').read_bytes())
assert execution['exit_code']==0 and not execution['stderr'] and (o/'RUN_COMPLETE').read_bytes()==b'completed\n'
assert summary['optimizer_updates']==summary['test_rows_evaluated']==summary['cos_writes']==0
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(r/'replay.py')==design['code_sha256']
cache=Path('/hy-tmp/Q-Attention/inputs/native-baseline/discovery/cache');y=np.load(cache/'valid-targets.npy',allow_pickle=False)
meta=json.loads(Path('/hy-tmp/Q-Attention/control/task-bottleneck-discovery-20261123/validation-cases.private.json').read_bytes())['metadata']
train=json.loads(Path('/hy-tmp/Q-Attention/control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
with np.load('/hy-tmp/Q-Attention/control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:fit=z['calibration'].copy()
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):
 g,st,ot=train[i];support[st,ot].add(g);seen.add(g)
support_sha=hashlib.sha256(json.dumps(sorted((list(pair),sorted(ids)) for pair,ids in support.items()),separators=(',',':')).encode()).hexdigest()
assert support_sha==design['frozen_support_sha256']
allow=np.ones((len(y),40),bool)
for i,(g,st,ot) in enumerate(meta):
 assert g==int(y[i])
 if (st,ot) in support:
  allow[i]=False;allow[i,0]=True
  for k in support[st,ot]|(set(range(40))-seen):allow[i,k]=True
pred={};hashes={};checks=[]
for mode,row in summary['results'].items():
 p=o/(mode+'-logits.private.npy');z=np.load(p,allow_pickle=False);assert z.shape==(19584,40) and np.isfinite(z).all();hashes[p.name]=sha(p)
 raw=z.argmax(1);typed=np.where(allow,z,-np.inf).argmax(1);pred[mode]=typed
 for scope,pp in [('raw',raw),('typed',typed)]:
  cm=np.bincount(y*40+pp,minlength=1600).reshape(40,40);assert np.array_equal(cm,np.array(row[scope]['confusion_matrix']))
  tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(40,float),where=den>0)
  assert abs(f[1:].mean()-row[scope]['positive39_macro_f1'])<1e-12
  assert np.trace(cm)==row[scope]['correct'] and cm.sum()==19584
  for k,idx in [('null_false_positive',(y==0)&(pp>0)),('positive_false_negative',(y>0)&(pp==0)),('positive_wrong_relation',(y>0)&(pp>0)&(pp!=y))]:assert int(idx.sum())==row[scope][k]
  checks.append(mode+'/'+scope)
 assert int(((raw!=y)&(typed==y)).sum())==row['repair'] and int(((raw==y)&(typed!=y)).sum())==row['damage']
strong=('relation_softmax','real_overlap','quantum','dephased')
common_wrong=np.logical_and.reduce([pred[m]!=y for m in strong]);common_fn=(y>0)&np.logical_and.reduce([pred[m]==0 for m in strong])
gold_allowed=allow[np.arange(len(y)),y]
baseline=pred['relation_softmax'];q=pred['quantum']
pairwise={}
for mode in strong:
 pp=pred[mode];repair=(baseline!=y)&(pp==y);damage=(baseline==y)&(pp!=y)
 pairwise[mode]=dict(repair=int(repair.sum()),damage=int(damage.sum()),net=int(repair.sum()-damage.sum()),
     repairs_softmax_positive_fn=int((repair&(y>0)&(baseline==0)).sum()),repairs_softmax_positive_wrong=int((repair&(y>0)&(baseline>0)).sum()),
     repairs_softmax_null_fp=int((repair&(y==0)).sum()),creates_positive_fn=int((damage&(y>0)&(pp==0)).sum()))
per_class=[];labels=list(summary['results']['quantum']['typed']['per_class'])
for k in range(1,40):
 target=y==k;rec=dict(label=labels[k]['label'],support=int(target.sum()),all_four_fn=int((target&common_fn).sum()),all_four_wrong=int((target&common_wrong).sum()),gold_excluded_by_fixed_rule=int((target&~gold_allowed).sum()))
 for mode in strong:rec[mode]=dict(correct=int((target&(pred[mode]==y)).sum()),false_negative=int((target&(pred[mode]==0)).sum()),wrong_relation=int((target&(pred[mode]>0)&(pred[mode]!=y)).sum()))
 per_class.append(rec)
residual=dict(scope='All four fixed nonuniform heads, same rebuilt VALID cache and unchanged type rule. Reused development evidence; no new method, oracle score, independent confirmation or causal attribution.',
    baseline='relation_softmax',models=list(strong),rows=len(y),positive_rows=int((y>0).sum()),gold_type_rule_exclusions=int((~gold_allowed).sum()),
    all_four_wrong=int(common_wrong.sum()),all_four_positive_fn=int(common_fn.sum()),all_four_positive_fn_with_gold_allowed=int((common_fn&gold_allowed).sum()),
    all_four_null_fp=int(((y==0)&np.logical_and.reduce([pred[m]>0 for m in strong])).sum()),
    softmax_false_negative_recoverable_by_other_existing_head=int(((baseline==0)&(y>0)&np.logical_or.reduce([pred[m]==y for m in strong if m!='relation_softmax'])).sum()),
    pairwise_vs_softmax=pairwise,per_class=per_class,
    caution='Common false negatives do not prove missing token binding, quantum advantage or representation impossibility. Rule-excluded golds are empirical support failures. Training budgets differ from the separate pooled reference.')
ret=json.loads((r/'resource-return.json').read_bytes());pid=ret['owner_pid'];resident=Path('/proc',str(pid)).exists()
if resident:
 current=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split();resident=current[0]!='Z' and current[19]==str(ret['owner_start_ticks'])
assert not resident
result=dict(summary=summary,residual=residual,execution=execution,resource_return=ret,verification=dict(status='PASS',confusion_reports=checks,checkpoint_outputs_sha256=hashes,resident_gone=True,exit_code=execution['exit_code'],run_complete_sha256=sha(o/'RUN_COMPLETE')))
(o/'verification.json').write_bytes((json.dumps(result['verification'],indent=2)+'\n').encode())
(o/'residual-summary.json').write_bytes((json.dumps(residual,indent=2)+'\n').encode())
print(json.dumps(result))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=35))
for name,value in [('summary.json',v['summary']),('residual-summary.json',v['residual']),('execution-verification.json',v)]:
    (r/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
(r/'RUN_COMPLETE').write_bytes(b'completed\n')
small={k:v['residual'][k] for k in v['residual'] if k!='per_class'}
print(json.dumps(dict(verification=v['verification']['status'],residual=small)))
