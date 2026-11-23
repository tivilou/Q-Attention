"""Posthoc arithmetic decomposition of the one already-sealed direction."""
import hashlib,json
from pathlib import Path
import numpy as np


def main(root):
    root=Path(root);private=root/'output/diagnostic.private.npz'
    assert hashlib.sha256(private.read_bytes()).hexdigest()=='e30ea3126cb9c07fdc1f2c88c10f45b0fb9760aebe06dc5e34a57fc801f6cde4'
    q=dict(np.load(private,allow_pickle=False));z=q['logits'];dz=q['logit_derivative'];y=q['labels'];cal=q['calibration']
    p=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);coeff=p.copy();coeff[np.arange(len(y)),y]-=1
    ce0=coeff[:,0]*dz[:,0];cepos=np.sum(coeff[:,1:]*dz[:,1:],axis=1)
    assert np.max(np.abs(ce0+cepos-q['ce_derivative']))<1e-10
    margin0=[];ties=0
    for i in range(len(y)):
        other=[r for r in range(40) if r!=int(y[i])];highest=max(z[i,r] for r in other)
        tied=[r for r in other if z[i,r]==highest];ties+=len(tied)>1
        competitor=max(tied,key=lambda r:dz[i,r])
        margin0.append((dz[i,0] if y[i]==0 else 0.)-(dz[i,0] if competitor==0 else 0.))
    margin0=np.array(margin0);marginpos=q['margin_derivative']-margin0;pred=z.argmax(-1)
    out={}
    for name,keep in dict(calibration=cal,assessment=~cal).items():
        masks=dict(all=keep,label_zero=keep&(y==0),label_positive=keep&(y!=0),
            positive_to_zero=keep&(y!=0)&(pred==0),zero_to_positive=keep&(y==0)&(pred!=0),
            wrong_positive=keep&(pred!=y)&(y!=0)&(pred!=0),native_correct=keep&(pred==y))
        table={}
        for key,m in masks.items():
            table[key]=dict(n=int(m.sum()),ce_class0_contribution=float(ce0[m].mean()),
                ce_positive_rows_contribution=float(cepos[m].mean()),
                margin_class0_contribution=float(margin0[m].mean()),
                margin_positive_rows_contribution=float(marginpos[m].mean()),
                class0_logit_derivative=float(dz[m,0].mean()))
        out[name]=table
    ga=q['gassessment'];d=q['direction'];inner0=float(np.vdot(ga[0],d[0]));innerpos=float(np.vdot(ga[1:],d[1:]))
    assert abs(inner0-out['assessment']['all']['ce_class0_contribution'])<1e-10
    assert abs(innerpos-out['assessment']['all']['ce_positive_rows_contribution'])<1e-10
    return dict(status='complete',scope='Posthoc exact first-order block decomposition of ONE previously frozen direction; not an alternative direction, finite update, causal training explanation or independent utility claim',
        input_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),partitions=out,
        exact_competitor_tie_rows=int(ties),class0_gradient_norm=float(np.linalg.norm(q['gcal'][0])),
        positive_rows_gradient_norm=float(np.linalg.norm(q['gcal'][1:])),
        assessment_block_inner_product_error=max(abs(inner0-out['assessment']['all']['ce_class0_contribution']),abs(innerpos-out['assessment']['all']['ce_positive_rows_contribution'])))


if __name__=='__main__':
    import sys
    print(json.dumps(main(sys.argv[1])))
