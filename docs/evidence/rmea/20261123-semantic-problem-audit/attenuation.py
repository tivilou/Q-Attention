"""Numerical all-interval check of one shared token attenuation on a pinned case."""
from pathlib import Path
import json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
CODE=r'''
from pathlib import Path
import numpy as np,json,math,hashlib,time
start=time.monotonic();base=Path('/hy-tmp/Q-Attention/control/semantic-problem-audit-20261123');out=Path('/hy-tmp/Q-Attention/control/semantic-native-detail-20261123/output')
assert not (base/'attenuation-summary.json').exists()
key='a8a180916f0013c7ea6b63b1';rows=json.loads((out/'cases.private.json').read_bytes());row=next(v for v in rows if v['case_id']==key)
assert row['semantic_gold_cue_eligible'] and row['all_token_single_row_upper_margin']<-.267
with np.load(out/'details.private.npz',allow_pickle=False) as z:a=z[key+'_alpha'].copy();u=z[key+'_utilities'].copy();logit=z[key+'_logits'].copy()
mu=(a*u).sum(1);bias=logit-mu;labels=json.loads(Path('/hy-tmp/Q-Attention/inputs/native-baseline/recovery-inputs/labels.json').read_bytes());g=labels.index(row['gold_label'])
types=json.loads(Path('/hy-tmp/Q-Attention/control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata'];truth=json.loads((base/'selection.private.json').read_bytes());record=next(v for v in truth if v['case_id']==key)
with np.load('/hy-tmp/Q-Attention/control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:fit=z['calibration'].copy()
support={};seen=set()
for i in np.flatnonzero(fit):
 c,st,ot=types[i];support.setdefault((st,ot),set()).add(c);seen.add(c)
allow=np.ones(40,bool);pair=tuple(types[record['row']][1:])
if pair in support:
 allow[:]=False;allow[0]=True
 for c in support[pair]|(set(range(40))-seen):allow[c]=True
assert allow[g] and np.max(a)<1 and np.min(a)>0
rivals=np.array([i for i in range(40) if allow[i] and i!=g]);old_top=int(rivals[np.argmax(logit[rivals])]);witnesses=[];near_degenerate=0;tests=0;equation_error=0.
for k in range(a.shape[1]):
 ag=a[g,k];ug=u[g,k];breaks=[0.,1.]
 for j in rivals:
  aj=a[j,k];uj=u[j,k];db=bias[g]-bias[j]
  c0=logit[g]-logit[j];c1=-db*(ag+aj)-mu[g]*aj-ag*ug+mu[j]*ag+aj*uj;c2=ag*aj*(db+ug-uj)
  if c2==0:
   roots=[] if c1==0 else [-c0/c1]
  else:
   disc=c1*c1-4*c2*c0
   if abs(disc)<1e-12:near_degenerate+=1
   if disc<0:roots=[]
   elif disc==0:roots=[-c1/(2*c2)]
   else:
    q=-.5*(c1+math.copysign(math.sqrt(disc),c1));roots=[q/c2,c0/q]
  breaks.extend(float(x) for x in roots if np.isfinite(x) and 0<x<1)
 breaks=sorted(set(breaks));candidates=[0.,1.]+[(x+y)/2 for x,y in zip(breaks,breaks[1:]) if y>x]
 found=None
 for lam in candidates:
  den=1-lam*a[:,k];assert den.min()>0
  changed=bias+(mu-lam*a[:,k]*u[:,k])/den;tests+=1
  for j in rivals:
   db=bias[g]-bias[j];ag=a[g,k];aj=a[j,k];ug=u[g,k];uj=u[j,k]
   c0=logit[g]-logit[j];c1=-db*(ag+aj)-mu[g]*aj-ag*ug+mu[j]*ag+aj*uj;c2=ag*aj*(db+ug-uj)
   exact=(c0+lam*c1+lam*lam*c2)/(den[g]*den[j]);equation_error=max(equation_error,abs(float(changed[g]-changed[j]-exact)))
  margin=float(changed[g]-changed[rivals].max())
  if margin>1e-8 and found is None:
   weighted=a.copy();weighted[:,k]*=1-lam;weighted/=weighted.sum(1,keepdims=True)
   independent=bias+(weighted*u).sum(1);assert np.max(abs(independent-changed))<1e-9
   top=int(rivals[np.argmax(changed[rivals])]);found=dict(token_index=k,lambda_value=lam,margin=margin,original_top=labels[old_top],new_top_rival=labels[top],target_score_change=float(changed[g]-logit[g]),original_top_score_drop=float(logit[old_top]-changed[old_top]),direct_recompute_max_error=float(np.max(abs(independent-changed))))
 if found is not None:witnesses.append(found)
assert equation_error<1e-8
report=dict(profile='fixed-obstructed-case-shared-one-token-attenuation-v1',case_id=key,tokens=a.shape[1],legal_rivals=len(rivals),
 original_gold_margin=float(logit[g]-logit[rivals].max()),target_only_upper_margin=row['all_token_single_row_upper_margin'],
 numerical_candidates_checked=tests,token_positions_with_verified_positive_witness=len(witnesses),first_witness=witnesses[0] if witnesses else None,
 near_degenerate_discriminants=near_degenerate,polynomial_vs_direct_max_abs=equation_error,seconds=time.monotonic()-start,
 scope='One fixed exposedTRAIN case selected because of its prior target-row obstruction. Numerical quadratic root partition, no parameter grid/fit. Positive witnesses checked directly; negative searches are not rigorous exact impossibility certificates. Same token attenuation in all40rows, same types, all allowed rivals checked. Target-aware existence only, no deployable rule, circuit reachability, quantum gain or L5.')
(base/'attenuation-witnesses.private.json').write_bytes(json.dumps(witnesses,indent=2).encode()+bytes([10]));(base/'attenuation-summary.json').write_bytes(json.dumps(report,indent=2).encode()+bytes([10]));print(json.dumps(report))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',CODE],timeout=30));(r/'attenuation-summary.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
