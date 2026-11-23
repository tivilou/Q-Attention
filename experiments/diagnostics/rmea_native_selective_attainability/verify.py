"""Recompute pointwise endpoints with independent generalized SciPy eigh."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.linalg import eigh
r=Path(sys.argv[1]);s=json.loads((r/'summary.json').read_bytes());seal=json.loads((r/'RUN_COMPLETE').read_bytes())
for n,h in seal['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
with np.load(r/'moments.private.npz',allow_pickle=False) as archive:
 v={k:archive[k] for k in archive.files}
ids=v['flip_ids'];y=v['labels'];z=v['logits'];correct=z.argmax(-1)==y
assert np.array_equal(ids,np.where(correct[0]!=correct[1])[0])
assert np.max(np.abs(z[:,~v['eligible']]-z[0,~v['eligible']]))<1e-8
M=np.empty((len(ids),2));upper=np.empty_like(M);error=0.
for i in range(len(ids)):
 for k in range(2):
  b=[]
  for c in range(40):
   w=eigh(v['A'][i,k,c],v['B'][i,k],eigvals_only=True,check_finite=True)
   b.append([w[0]+v['bias'][c],w[-1]+v['bias'][c]])
  b=np.array(b);delta=float(np.max(np.abs(b-v['bounds'][i,k])));error=max(error,delta)
  assert delta<max(1e-7,float(v['uncertainty'][i,k].max()))
  low=b[:,0].copy();low[y[ids[i]]]=-np.inf;M[i,k]=b[y[ids[i]],1]-max(low)
  adjusted=b[:,0]-v['uncertainty'][i,k];adjusted[y[ids[i]]]=-np.inf
  upper[i,k]=b[y[ids[i]],1]+v['uncertainty'][i,k,y[ids[i]]]-max(adjusted)
assert np.max(np.abs(M-v['margins'][:,:,0]),initial=0)<1e-6
assert np.max(np.abs(upper-v['margins'][:,:,1]),initial=0)<1e-6
damage=correct[0,ids];repair=~damage;tol=1e-7
dw=damage&(upper[:,1]<-tol);rw=repair&(upper[:,0]<-tol)
assert np.array_equal(dw,damage&(v['margins'][:,1,1]<-tol)) and np.array_equal(rw,repair&(v['margins'][:,0,1]<-tol)), 'threshold disagreement: unresolved'
assert np.all(M[dw,1]<-tol) and np.all(M[rw,0]<-tol)
for key,m in [('damage',damage),('repair',repair),('damage_witness',dw),('repair_witness',rw)]:assert np.bincount(y[ids[m]],minlength=40).tolist()==s['per_class'][key]
assert (int(damage.sum()),int(repair.sum()),int(dw.sum()),int(rw.sum()))==(s['damaged'],s['repaired'],s['damage_witnesses'],s['repair_witnesses'])
gate='PASS_BIDIRECTIONAL_EXISTENCE_ONLY' if dw.any() and rw.any() else 'FAIL_CLOSE_BIDIRECTIONAL_PROFILE';assert s['gate']==gate
print(json.dumps(dict(status='PASS',gate=gate,all40class_endpoint_and_count_recompute=True,maximum_endpoint_error=error,numerical_not_interval_certificate=True)))
