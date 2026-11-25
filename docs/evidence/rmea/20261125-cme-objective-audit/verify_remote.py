"""Independent math-only row arithmetic; NumPy is used only to read existing NPZ."""
from pathlib import Path
import hashlib,json,math,time,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
start=time.monotonic()
root=Path('/hy-tmp/Q-Attention');folder=root/'control/distinct-token-evidence-20261125/output'
path=folder/'diagnostic.private.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307'
models=['quantum_distinct','dephased_distinct','gaussian_distinct','quantum_permuted_distinct','quantum_collision_mass']
with np.load(path,allow_pickle=False) as d:
    indices=d['assessment_indices'].tolist();labels=d['labels'].tolist();flags=d['calibration'].tolist()
    y=[int(labels[i]) for i in indices]
    logits={m:d[m+'_logits'].tolist() for m in models}
assert len(y)==4096 and set(y)==set(range(40))
fit_support=[0]*40
for label,flag in zip(labels,flags):
    if flag:fit_support[int(label)]+=1
assert sum(fit_support)==8192
argmax=lambda row:max(range(40),key=lambda i:row[i])
qpred=[argmax(z) for z in logits[models[0]]];gpred=[argmax(z) for z in logits[models[2]]]
fields=['ce','brier','truth_probability','probability_square_sum','max_probability','entropy','ce_gradient_norm','brier_gradient_norm']
def empty():return {'rows':0,**{k:0. for k in fields}}
def add(acc,values):
    acc['rows']+=1
    for k in fields:acc[k]+=values[k]
def finish(acc):
    n=acc['rows'];return dict(rows=n,**{k:(acc[k]/n if n else None) for k in fields})
group_names=[f'{"positive" if positive else "null"}_q{q}_g{g}' for positive in [False,True] for q in [False,True] for g in [False,True]]
# Use integer q/g names to match the semantic schema, with separate grouping logic.
group_names=[n.replace('False','0').replace('True','1') for n in group_names]
result={}
max_identity_residual=0.
for model in models:
    total=empty();classes=[empty() for _ in range(40)];groups={n:empty() for n in group_names};high=empty();ratio_sum=0.
    cm=[[0]*40 for _ in range(40)]
    for i,(row,truth) in enumerate(zip(logits[model],y)):
        prediction=argmax(row);cm[truth][prediction]+=1
        maximum=max(row);shift=[z-maximum for z in row]
        exps=[math.exp(x) for x in shift];den=math.fsum(exps);p=[x/den for x in exps]
        ce=math.log(den)-shift[truth];square=math.fsum(x*x for x in p)
        errors=[x-int(k==truth) for k,x in enumerate(p)]
        brier=math.fsum(x*x for x in errors)
        c=square-p[truth]
        gb=math.sqrt(math.fsum((2*x*(errors[k]-c))**2 for k,x in enumerate(p)))
        gc=math.sqrt(brier)
        values=dict(ce=ce,brier=brier,truth_probability=p[truth],probability_square_sum=square,
            max_probability=max(p),entropy=math.log(den)-math.fsum(x*z for x,z in zip(p,shift)),
            ce_gradient_norm=gc,brier_gradient_norm=gb)
        max_identity_residual=max(max_identity_residual,abs(brier-(1+square-2*p[truth])))
        add(total,values);add(classes[truth],values)
        name=('positive' if truth else 'null')+'_q'+str(int(qpred[i]==truth))+'_g'+str(int(gpred[i]==truth))
        add(groups[name],values)
        if prediction!=truth and max(p)>=.9:
            add(high,values);ratio_sum+=gb/gc
    per_class=[dict(label=c,fit_support=fit_support[c],**finish(acc)) for c,acc in enumerate(classes)]
    f1=[]
    for c in range(40):
        den=sum(cm[c])+sum(row[c] for row in cm)
        f1.append(2*cm[c][c]/den if den else 0.)
    result[model]=dict(overall=finish(total),class_macro_brier=math.fsum(a['brier'] for a in per_class)/40,
        original_macro_f1=math.fsum(f1)/40,correct=sum(cm[c][c] for c in range(40)),
        confusion_sha256=hashlib.sha256(json.dumps(cm,separators=(',',':')).encode()).hexdigest(),
        by_class=per_class,paired_groups={name:finish(acc) for name,acc in groups.items()},
        high_confidence_wrong=dict(**finish(high),mean_brier_to_ce_gradient_ratio=(ratio_sum/high['rows'] if high['rows'] else None)))
assert max_identity_residual<2e-15
print(json.dumps(dict(models=result,fit_support=fit_support,rows=4096,max_brier_identity_residual=max_identity_residual,seconds=time.monotonic()-start),separators=(',',':')))
