"""Independent metrics/gate recomputation from immutable private predictions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--reference',type=Path)
a=ap.parse_args();r=a.output;s=json.loads((r/'summary.json').read_bytes())
for n,h in json.loads((r/'RUN_COMPLETE').read_bytes())['files'].items():
    assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
v=dict(np.load(r/'predictions.private.npz',allow_pickle=False));z=v['logits'];y=v['labels']
assert z.shape==(4,19584,40) and y.shape==(19584,) and np.isfinite(z).all()
rows={};max_error=0.
for k,name in enumerate(('native','frequency','balanced','common')):
    pred=z[k].argmax(-1);cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(40),where=den!=0)
    zz=z[k]-z[k].max(-1,keepdims=True);ce=np.log(np.exp(zz).sum(-1))-zz[np.arange(len(y)),y]
    m=dict(ce=float(ce.mean()),class0_ce=float(ce[y==0].mean()),positive_ce=float(ce[y!=0].mean()),
        positive_macro_f1=float(f1[1:].mean()),accuracy=float(np.mean(pred==y)))
    row=s['results'][name];assert np.array_equal(cm,row['confusion_matrix'])
    for key,value in m.items():
        error=abs(row[key]-value);assert error<1e-12;max_error=max(max_error,error)
    old=z[0].argmax(-1)
    for key,value in dict(repaired=int(((old!=y)&(pred==y)).sum()),damaged=int(((old==y)&(pred!=y)).sum()),
        changed=int((old!=pred).sum()),false_negative=int(((y!=0)&(pred==0)).sum()),false_positive=int(((y==0)&(pred!=0)).sum())).items():assert row[key]==value
    rows[name]=m
c=rows['common'];n=rows['native'];t=1e-8
gates=dict(class0_ce_decreases=c['class0_ce']<n['class0_ce']-t,positive_ce_decreases=c['positive_ce']<n['positive_ce']-t,
    macro_beats_native=c['positive_macro_f1']>n['positive_macro_f1']+t,
    macro_beats_balanced=c['positive_macro_f1']>rows['balanced']['positive_macro_f1']+t,
    macro_beats_frequency=c['positive_macro_f1']>rows['frequency']['positive_macro_f1']+t)
assert gates==s['gates'] and s['gate']==('PASS_FIXED_STEP_ONLY' if all(gates.values()) else 'FAIL_CLOSE_EXACT_STEP')
assert s['eta']==.05 and s['optimizer_updates']==0 and s['source_checkpoint_unchanged']
print(json.dumps(dict(status='PASS',gate=s['gate'],all_metrics_confusions_changes_and_five_gates_recomputed=True,max_metric_error=max_error)))
