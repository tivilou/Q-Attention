"""Read frozen gen98 logits only; emit all-arm aggregate objective diagnostics."""
from pathlib import Path
import os,time,json,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
start=time.monotonic()
root=Path('/hy-tmp/Q-Attention')
folder=root/'control/distinct-token-evidence-20261125/output'
npz=folder/'diagnostic.private.npz'
cohort_path=root/'control/staged-bottleneck-design-20261125/cohort.private.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(npz)=='3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307'
assert sha(cohort_path)=='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0'
cohort=json.loads(cohort_path.read_bytes())
models=('quantum_distinct','dephased_distinct','gaussian_distinct','quantum_permuted_distinct','quantum_collision_mass')
reference=json.loads((folder/'summary.json').read_bytes())
with np.load(npz,allow_pickle=False) as d:
    ix=d['assessment_indices'].copy();fit=np.flatnonzero(d['calibration']);labels=d['labels'].copy()
    assert ix.tolist()==cohort['assessment'] and set(map(int,fit))==set(cohort['fit'])
    assert len(ix)==4096 and len(fit)==8192
    y=labels[ix];assert np.array_equal(np.unique(y),np.arange(40))
    logits={m:d[m+'_logits'].astype(np.float64) for m in models}
    assert all(a.shape==(4096,40) and np.isfinite(a).all() for a in logits.values())
    assert not set(cohort['selected_groups'][str(i)] for i in fit).intersection(cohort['selected_groups'][str(i)] for i in ix)
fit_support=np.bincount(labels[fit],minlength=40).tolist()
pred={m:logits[m].argmax(-1) for m in models}
qok=pred[models[0]]==y;gok=pred[models[2]]==y
group_id=4*(y>0).astype(int)+2*qok.astype(int)+gok.astype(int)
names={4*positive+2*q+g:f'{"positive" if positive else "null"}_q{q}_g{g}' for positive in [0,1] for q in [0,1] for g in [0,1]}
fields=('ce','brier','truth_probability','probability_square_sum','max_probability','entropy','ce_gradient_norm','brier_gradient_norm')
rows={}
def summarize(values,mask):
    n=int(mask.sum())
    return dict(rows=n,**{k:(float(values[k][mask].mean()) if n else None) for k in fields})
for m,z in logits.items():
    shifted=z-z.max(-1,keepdims=True);exp=np.exp(shifted)
    logp=shifted-np.log(exp.sum(-1,keepdims=True));p=np.exp(logp)
    truth=p[np.arange(4096),y];square=(p*p).sum(-1)
    delta=p.copy();delta[np.arange(4096),y]-=1
    brier=(delta*delta).sum(-1)
    gradient=2*p*(delta-(p*delta).sum(-1,keepdims=True))
    values=dict(ce=-logp[np.arange(4096),y],brier=brier,truth_probability=truth,
        probability_square_sum=square,max_probability=p.max(-1),entropy=-(p*logp).sum(-1),
        ce_gradient_norm=np.linalg.norm(delta,axis=-1),brier_gradient_norm=np.linalg.norm(gradient,axis=-1))
    assert np.max(abs(brier-(1+square-2*truth)))<2e-15
    cm=np.bincount(40*y+pred[m],minlength=1600).reshape(40,40)
    assert cm.tolist()==reference['results'][m]['confusion_matrix']
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den>0)
    assert abs(float(f1.mean())-reference['results'][m]['all40_macro_f1'])<1e-14
    by_class=[dict(label=c,fit_support=fit_support[c],**summarize(values,y==c)) for c in range(40)]
    high=(pred[m]!=y)&(p.max(-1)>=.9)
    ratio=values['brier_gradient_norm']/np.maximum(values['ce_gradient_norm'],1e-300)
    rows[m]=dict(overall=summarize(values,np.ones(4096,dtype=bool)),
        class_macro_brier=float(np.mean([a['brier'] for a in by_class])),
        original_macro_f1=float(f1.mean()),correct=int(np.trace(cm)),
        confusion_sha256=hashlib.sha256(json.dumps(cm.tolist(),separators=(',',':')).encode()).hexdigest(),
        by_class=by_class,paired_groups={names[g]:summarize(values,group_id==g) for g in range(8)},
        high_confidence_wrong=dict(**summarize(values,high),mean_brier_to_ce_gradient_ratio=(float(ratio[high].mean()) if high.any() else None)))
q=rows[models[0]]['overall'];g=rows[models[2]]['overall']
out=dict(status='PASS_EXISTING_GEN98_OBJECTIVE_DIAGNOSTIC',npz_sha256=sha(npz),cohort_sha256=sha(cohort_path),
    row_count=4096,fit_support=fit_support,models=rows,
    q_minus_g_decomposition=dict(brier=q['brier']-g['brier'],probability_square_sum=q['probability_square_sum']-g['probability_square_sum'],minus_two_truth_probability=-2*(q['truth_probability']-g['truth_probability'])),
    seconds=time.monotonic()-start,new_head_forwards=0,new_updates=0,new_gpu_allocations=0,
    new_model_feature_text_reads=0,server_writes=0,development_only=True)
print(json.dumps(out,separators=(',',':')))
