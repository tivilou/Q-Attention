"""R0: separate positive-family posterior mass from positive-label identity."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 q=p.with_suffix('.pending');q.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());q.replace(p)
def main():
 ap=argparse.ArgumentParser()
 for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes());r=a.root
 assert sha(__file__)==cfg['code_sha256']
 for n,h in cfg['input_hashes'].items():assert sha(r/n)==h,n
 assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic()
 meta=json.loads((r/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
 vm=json.loads((r/'control/task-bottleneck-discovery-20261123/validation-cases.private.json').read_bytes())['metadata']
 with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as q:fit=q['calibration'].copy()
 support=defaultdict(set);seen=set()
 for i in np.flatnonzero(fit):g,s,o=meta[i];support[s,o].add(g);seen.add(g)
 allowed=np.ones((len(vm),40),bool)
 for i,(_,s,o) in enumerate(vm):
  if (s,o) in support:
   allowed[i]=False;allowed[i,0]=True
   for g in support[s,o]|(set(range(40))-seen):allowed[i,g]=True
 y=np.load(r/'inputs/native-baseline/discovery/cache/valid-targets.npy',allow_pickle=False)
 selection=json.loads((r/'control/full-model-error-structure-20261123/semantic/selection.private.json').read_bytes())
 semantics={q['case_id']:q for q in json.loads((r/'control/full-model-error-structure-20261123/semantic/semantic-results.private.json').read_bytes())}
 labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
 modes=cfg['modes'];zs={m:np.load(r/'control/historical-type-replay-20261123/output'/ (m+'-logits.private.npy'),allow_pickle=False).astype(float) for m in modes}
 preds={m:np.where(allowed,z,-np.inf).argmax(-1) for m,z in zs.items()}
 common=(y>0)&np.logical_and.reduce([p==0 for p in preds.values()]);legal=allowed[np.arange(len(y)),y]
 assert int(common.sum())==1155 and int((common&legal).sum())==1070
 out={};case_rows=[];residual=0.
 def stats(e,mask):
  v=e[mask];return dict(rows=len(v),positive_family_mass_exceeds_null=int((v>0).sum()),family_logodds_quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist() if len(v) else [])
 for cursor,m in enumerate(modes,1):
  z=zs[m];positive=np.where(allowed[:,1:],z[:,1:],-np.inf);mx=positive.max(-1)
  e=mx+np.log(np.exp(positive-mx[:,None]).sum(-1))-z[:,0]
  top=positive.argmax(-1)+1;gap=mx-z[:,0]
  offset=e-gap;assert np.all(offset>=-1e-10) and np.all(offset<=np.log(39)+1e-10)
  # Independent softmax-sum identity; not a changed decision rule or a fitted threshold.
  zz=np.where(allowed,z,-np.inf);p=np.exp(zz-zz.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
  direct=np.log(p[:,1:].sum(-1))-np.log(p[:,0]);err=float(np.abs(direct-e).max());residual=max(residual,err);assert err<1e-10
  out[m]=dict(common_legal_fn=stats(e,common&legal),common_legal_fn_gold_top_positive=stats(e,common&legal&(top==y)),
   own_positive_fn=stats(e,(y>0)&(preds[m]==0)),own_correct_null=stats(e,(y==0)&(preds[m]==0)),
   own_null_false_positive=stats(e,(y==0)&(preds[m]!=0)),own_correct_positive=stats(e,(y>0)&(preds[m]==y)))
  for rec in selection:
   i=rec['row'];s=semantics[rec['case_id']]
   case_rows.append(dict(case_id=rec['case_id'],mode=m,family=rec['family'],gold=rec['gold'],prediction=labels[int(preds[m][i])],
    semantic_supported=bool(s['confident_consensus'] and s['consensus']==rec['gold']),
    best_positive_minus_null=float(gap[i]),positive_family_minus_null=float(e[i]),positive_identity_ambiguity=float(offset[i])))
  save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='coarse_posterior_observation',current=cursor,total=len(modes),unit='head',observed_at=time.time(),heartbeat_at=time.time()))
 save(a.output/'cases.private.json',case_rows)
 report=dict(profile=cfg['profile'],attempt=a.attempt,seconds=time.monotonic()-start,rows=len(y),results=out,
  coarse_identity_max_abs=residual,model_forwards=0,optimizer_updates=0,test_rows=0,cos_writes=0,
  interpretation='Known probability coarse-graining. Family mass > null does not mean correct relation or correct macro-optimal decision. No new prediction, threshold fit or promotion; all original predictions retained.',
  scope='R0 descriptive arithmetic on reused complete-model VALID outputs; no causal or independent generalization/quantum/L5 claim.',
  private_cases_sha256=sha(a.output/'cases.private.json'))
 save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n');print(json.dumps(report))
if __name__=='__main__':main()
