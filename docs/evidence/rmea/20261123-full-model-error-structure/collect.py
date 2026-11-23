"""R0: recompute fixed-model metrics and describe error/support structure on compute."""
from pathlib import Path
import json
import sys

r = Path(__file__).resolve().parent
sys.path.insert(0, 'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops

CODE = r'''
from pathlib import Path
from collections import defaultdict, Counter
import hashlib,json
import numpy as np
root=Path('/hy-tmp/Q-Attention');r=root/'control/full-model-error-structure-20261123';o=r/'output'
prior=root/'control/historical-type-replay-20261123/output';cache=root/'inputs/native-baseline/discovery/cache'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((o/'summary.json').read_bytes());ex=json.loads((r/'execution.json').read_bytes());cfg=json.loads((r/'design.json').read_bytes())
assert ex['exit_code']==0 and not ex['stderr'] and (o/'RUN_COMPLETE').read_bytes()==b'completed\n'
assert summary['optimizer_updates']==summary['test_rows_evaluated']==summary['cos_writes']==0
assert sha(r/'replay.py')==cfg['code_sha256']
modes=list(summary['results']);strong=['relation_softmax','real_overlap','quantum','dephased']
labels=json.loads((root/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
ys={s:np.load(cache/(s+'-targets.npy'),allow_pickle=False) for s in ('train','valid')}
zs={s:{m:np.load((o if s=='train' else prior)/(m+'-logits.private.npy'),allow_pickle=False) for m in modes} for s in ys}
pred={s:{m:z.argmax(1) for m,z in rows.items()} for s,rows in zs.items()}
oldvalid=json.loads((prior/'summary.json').read_bytes());checks=[];hashes={};max_residual=0.
for split,y in ys.items():
 for mode,z in zs[split].items():
  assert z.shape==(len(y),40) and np.isfinite(z).all()
  pp=pred[split][mode];cm=np.bincount(y*40+pp,minlength=1600).reshape(40,40)
  row=(summary if split=='train' else oldvalid)['results'][mode]['raw']
  assert np.array_equal(cm,row['confusion_matrix'])
  f=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(40,float),where=(cm.sum(0)+cm.sum(1))>0)
  residual=abs(float(f[1:].mean())-row['positive39_macro_f1']);max_residual=max(max_residual,residual);assert residual<1e-12
  assert cm.trace()==row['correct']
  for k,mask in [('null_false_positive',(y==0)&(pp>0)),('positive_false_negative',(y>0)&(pp==0)),('positive_wrong_relation',(y>0)&(pp>0)&(pp!=y))]:assert int(mask.sum())==row[k]
  if split=='train':
   zz=z.astype(float);zz-=zz.max(1,keepdims=True);ce=float(np.mean(np.log(np.exp(zz).sum(1))-zz[np.arange(len(y)),y]))
   assert abs(ce-row['ce'])<1e-12
   hashes[mode]=sha(o/(mode+'-logits.private.npy'))
  checks.append(split+'/'+mode)
# Reuse exactly the already fixed empirical type rule, including retained unseen classes.
tm=json.loads((root/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
vm=json.loads((root/'control/task-bottleneck-discovery-20261123/validation-cases.private.json').read_bytes())['metadata']
with np.load(root/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as f:fit=f['calibration'].copy()
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):
 g,st,ot=tm[i];assert g==ys['train'][i];support[st,ot].add(g);seen.add(g)
support_sha=hashlib.sha256(json.dumps(sorted((list(k),sorted(v)) for k,v in support.items()),separators=(',',':')).encode()).hexdigest()
assert support_sha=='d4c1100df6d306a3fae48d66f7907dfbe2a39d1b30843abb359ba32de7053186'
def allowed(meta):
 a=np.ones((len(meta),40),bool)
 for i,(g,st,ot) in enumerate(meta):
  if (st,ot) in support:
   a[i]=False;a[i,0]=True
   for k in support[st,ot]|(set(range(40))-seen):a[i,k]=True
 return a
at=allowed(tm);av=allowed(vm)
typed={'train':{m:np.where(at,z[:8192],-np.inf).argmax(1) for m,z in zs['train'].items()},
       'valid':{m:np.where(av,z,-np.inf).argmax(1) for m,z in zs['valid'].items()}}
common_raw={s:(y>0)&np.logical_and.reduce([pred[s][m]==0 for m in strong]) for s,y in ys.items()}
common=(ys['valid']>0)&np.logical_and.reduce([typed['valid'][m]==0 for m in strong])
gold_allowed=av[np.arange(len(av)),ys['valid']]
assert common.sum()==1155 and (common&gold_allowed).sum()==1070
per_class=[]
for k in range(1,40):
 row=dict(label=labels[k],train_support=int((ys['train']==k).sum()),valid_support=int((ys['valid']==k).sum()),
          valid_shared_typed_fn=int(((ys['valid']==k)&common).sum()))
 for s,y in ys.items():
  take=y==k;row[s]={m:dict(correct=int(((pred[s][m]==k)&take).sum()),false_negative=int(((pred[s][m]==0)&take).sum())) for m in strong}
 per_class.append(row)
# Gold-aware diagnostic only: how often the correct positive label already outranks other positive labels.
rank={}
y=ys['valid'];idx=np.flatnonzero(common);yg=y[idx]
for m in strong:
 z=zs['valid'][m][idx].astype(float);legal=av[idx].copy();legal[:,0]=False
 positive=np.where(legal,z,-np.inf);gz=z[np.arange(len(idx)),yg]
 ranks=1+(positive>gz[:,None]).sum(1)
 rank[m]=dict(rows=len(idx),gold_allowed=int(gold_allowed[idx].sum()),
  gold_top_positive=int(((ranks==1)&gold_allowed[idx]).sum()),gold_top3_positive=int(((ranks<=3)&gold_allowed[idx]).sum()),
  gold_minus_null_quantiles=np.quantile(gz-z[:,0],[0,.25,.5,.75,1]).tolist())
# Align the entire old selected32, never select a new favorable subset.
selection=json.loads((root/'control/semantic-problem-audit-20261123/selection.private.json').read_bytes())
assert len(selection)==32
confirmed={'76ddecf4a92c7b3f6ef01155','aad7dcb5fbb1006979de2555','a8a180916f0013c7ea6b63b1','b6cec6caebd3888447c9ac74','1139bb04a87a7d9d86b251ef'}
alignment=[]
for rec in selection:
 i=rec['row'];g=int(ys['train'][i]);assert labels[g]==rec['gold']
 alignment.append(dict(case_id=rec['case_id'],old_family=rec['family'],gold=labels[g],two_readers_supported_missed_gold=rec['case_id'] in confirmed,
    full_raw_predictions={m:labels[int(pred['train'][m][i])] for m in strong},full_typed_predictions={m:labels[int(typed['train'][m][i])] for m in strong}))
alignment_counts={}
for family in sorted({q['old_family'] for q in alignment}):
 rows=[q for q in alignment if q['old_family']==family]
 alignment_counts[family]=dict(rows=len(rows),full_typed_correct={m:sum(q['full_typed_predictions'][m]==q['gold'] for q in rows) for m in strong},
  still_all_four_null=sum(all(q['full_typed_predictions'][m]=='no_relation' for m in strong) for q in rows))
confirmed_summary={m:sum(q['two_readers_supported_missed_gold'] and q['full_typed_predictions'][m]==q['gold'] for q in alignment) for m in strong}
# Descriptive support exposure, not a causal or gain estimate. Hashes/row contents stay private.
meta=json.loads((cache/'metadata.json').read_bytes());assert len(meta['train'])==len(ys['train']) and len(meta['valid'])==len(y)
def keys(q):
 tokens=q['tokens'];s=tuple(q['subject']);t=tuple(q['object']);assert len(tokens)==q['words'] and 0<=s[0]<s[1]<=len(tokens) and 0<=t[0]<t[1]<=len(tokens)
 sub=tuple(w.casefold() for w in tokens[s[0]:s[1]]);obj=tuple(w.casefold() for w in tokens[t[0]:t[1]])
 masked=[]
 for j,w in enumerate(tokens):
  if j==s[0]:masked.append('<SUBJECT>')
  if j==t[0]:masked.append('<OBJECT>')
  if not (s[0]<=j<s[1] or t[0]<=j<t[1]):masked.append(w.casefold())
 return (sub,obj),tuple(masked)
exposure=dict(status='NOT_ESTIMATED',reason='Cache metadata contains tokens/spans only for first2 rows. Full TRAIN lexical support is unavailable in this cache; no partial8192 support presented as full-training exposure.',
  metadata_rows_with_tokens={split:sum('tokens' in q for q in meta[split]) for split in ('train','valid')})
length_groups={}
for split in ('train','valid'):
 yy=ys[split];ll=np.array([q['words'] for q in meta[split]])
 shared=common_raw[split];rows=[]
 for lo,hi in [(0,20),(21,40),(41,80),(81,1000000)]:
  pos=(yy>0)&(ll>=lo)&(ll<=hi)
  rows.append(dict(word_length_range=[lo,hi],positive_rows=int(pos.sum()),shared_raw_fn=int((shared&pos).sum()),
       shared_raw_fn_rate=float((shared&pos).sum()/pos.sum()) if pos.any() else None))
 length_groups[split]=rows
structure=dict(scope='Descriptive reused development and complete-model current-cache replay. No new fit, selection, TEST score or quantum attribution; exposure bins are not causal.',
  raw_common_fn={s:dict(positive_rows=int((ys[s]>0).sum()),all_four_fn=int(common_raw[s].sum())) for s in ys},
  valid_typed_common_fn=int(common.sum()),valid_typed_common_fn_gold_allowed=int((common&gold_allowed).sum()),
  class_support_and_errors=per_class,gold_rank_within_fixed_allowed_positives=rank,training_support_exposure=exposure,length_groups=length_groups,
  old_semantic32_alignment=alignment,old_semantic32_counts=alignment_counts,old_five_supported_now_full_typed_correct=confirmed_summary,
  cautions=['Old32 are TRAIN examples from a four-epoch checkpoint, not complete-model development errors. Full models have seen these TRAIN rows; repair here demonstrates budget/cohort mismatch, not generalization.',
            'Full-model checkpoints were selected on historical VALID; current h differs. No exact original fitting-loss or full-cache drift bound.',
            'Gold ranks, shared errors and support-exposure rates do not by themselves yield a deployable correction or causal mechanism.'])
ret=json.loads((r/'resource-return.json').read_bytes());resident=Path('/proc',str(ret['owner_pid'])).exists();assert not resident
verification=dict(status='PASS',independent_confusion_checks=checks,max_macro_residual=max_residual,train_logits_sha256=hashes,
  old_selection_sha256=sha(root/'control/semantic-problem-audit-20261123/selection.private.json'),resident_gone=True,exit_code=0,run_complete_sha256=sha(o/'RUN_COMPLETE'))
for n,v in [('structure.json',structure),('verification.json',verification)]:
 p=o/n;data=(json.dumps(v,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==data,n
 else:p.write_bytes(data)
print(json.dumps(dict(summary=summary,structure=structure,verification=verification,execution=ex,resource_return=ret)))
'''

v = json.loads(ops.remote([ops.PY, '-B', '-c', CODE], timeout=60))
for n, key in [('summary.json','summary'),('structure.json','structure'),('verification.json','verification'),('execution.json','execution'),('resource-return.json','resource_return')]:
    data=(json.dumps(v[key], indent=2)+'\n').encode()
    p=r/n
    if p.exists():
        assert p.read_bytes()==data,n
    else:
        p.write_bytes(data)
(r/'RUN_COMPLETE').write_bytes(b'completed\n')
st=v['structure']
print(json.dumps(dict(verification=v['verification']['status'], raw_common_fn=st['raw_common_fn'], rank=st['gold_rank_within_fixed_allowed_positives'],
    semantic_counts=st['old_semantic32_counts'],five_supported=st['old_five_supported_now_full_typed_correct'],exposure=st['training_support_exposure']),indent=2))
