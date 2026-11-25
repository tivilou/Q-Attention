"""Read already-produced TRAIN logits on their host; return compact counts only."""
from pathlib import Path
import hashlib
import json
import sys

r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
sources={r.name:['quantum','dephased','gaussian_base'],
    'context-spectral-flow-20261125':['quantum','dephased'],
    'staged-bottleneck-design-20261125':['gaussian_base'],
    'connected-value-pooling-20261125':['gaussian_covariance']}
request=[]
for name,modes in sources.items():
    summary=json.loads((r.parent/name/'summary.json').read_bytes())
    request.append(dict(name=name,modes=modes,sha256=summary['private_diagnostic_sha256']))
code=r'''from pathlib import Path
import hashlib,json,sys
import numpy as np
request=json.load(sys.stdin);rows={};preds={};reference=None
for item in request:
 p=Path('/hy-tmp/Q-Attention/control')/item['name']/'output/diagnostic.private.npz'
 assert p.stat().st_size<40*1024**2 and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 with np.load(p,allow_pickle=False) as v:
  ids=v['assessment_indices'];y=v['labels'][ids]
  assert len(ids)==4096 and all(np.bincount(y,minlength=40)>0)
  if reference is None:reference=(ids.copy(),y.copy())
  else:assert np.array_equal(ids,reference[0]) and np.array_equal(y,reference[1])
  positive=y>0
  for mode in item['modes']:
   z=v[mode+'_logits'];assert z.shape==(4096,40) and np.isfinite(z).all()
   pred=z.argmax(-1);forced=z[:,1:].argmax(-1)+1
   good=forced==y
   cm=np.bincount(39*(y[positive]-1)+(forced[positive]-1),minlength=39**2).reshape(39,39)
   den=cm.sum(0)+cm.sum(1)
   f1=np.divide(2*np.diag(cm),den,out=np.zeros(39),where=den!=0)
   key=('adjusted/' if item['name']==request[0]['name'] else 'reused/')+mode
   rows[key]=dict(positive_items=int(positive.sum()),
    actual_positive_correct=int((positive&(pred==y)).sum()),
    forced_positive_correct=int((positive&good).sum()),
    null_blocked_but_positive_rank_correct=int((positive&(pred==0)&good).sum()),
    wrong_positive_label_even_without_null=int((positive&~good).sum()),
    conditional_positive39_macro=float(f1.mean()),conditional_f1=f1.tolist())
   preds[key]=good[positive]
q=preds['reused/quantum'];pairs={}
for key,v in preds.items():
 if key!='reused/quantum':pairs['nativeQ_vs_'+key]=dict(Q_correct_other_wrong=int((q&~v).sum()),other_correct_Q_wrong=int((v&~q).sum()))
print(json.dumps(dict(status='completed',new_fits=0,new_forward_passes=0,new_heldout_reads=0,cos_writes=0,
 result_scope='Post-outcome descriptive decomposition of already-produced fixed TRAIN4096 logits; conditioned on gold positives, not deployable inference or primary gate',
 rows=rows,paired_positive_ranking_counts=pairs)))
'''
out=json.loads(ops.remote(['/hy-tmp/Q-Attention/env/bin/python','-B','-c',code],data=json.dumps(request).encode(),timeout=30))
out['input_private_diagnostic_sha256']={v['name']:v['sha256'] for v in request}
out['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(r/'positive-label-ranking.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(rows={k:{a:b for a,b in v.items() if a!='conditional_f1'} for k,v in out['rows'].items()},pairs=out['paired_positive_ranking_counts'])))
