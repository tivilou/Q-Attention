"""One calibration-fit no-relation offset and a restricted accuracy oracle."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path('/hy-tmp/Q-Attention/control/native-ordered-mass-value-20261123/output')


def describe(z,y):
    prediction=z.argmax(1);wrong=prediction!=y
    cm=np.bincount(y*40+prediction,minlength=1600).reshape(40,40)
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),den,out=np.zeros(40),where=den>0)
    shifted=z-z.max(1,keepdims=True)
    ce=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y]
    return dict(items=len(y),accuracy=float((~wrong).mean()),errors=int(wrong.sum()),
        ce=float(ce.mean()),positive39_macro_f1=float(f1[1:].mean()),
        positive_to_zero=int(((y>0)&(prediction==0)).sum()),
        zero_to_positive=int(((y==0)&(prediction>0)).sum()),
        wrong_positive_relation=int(((y>0)&(prediction>0)&wrong).sum()))


def main():
    p=ROOT/'diagnostic.private.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='ff920146542e7af85f90df5ececfa62f0742be4e90f0db4de8645946e043d833'
    with np.load(p,allow_pickle=False) as a:
        z=a['logits'];y=a['labels'];fit=a['calibration'];assessment=~fit
    def derivative(b):
        x=z[fit].copy();x[:,0]+=b;x-=x.max(1,keepdims=True)
        ex=np.exp(x)
        return float((ex[:,0]/ex.sum(1)).mean()-(y[fit]==0).mean())
    lo=-64.;hi=64.;assert derivative(lo)<0<derivative(hi)
    for _ in range(100):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    offset=(lo+hi)/2
    zz=z[assessment];yy=y[assessment];modified=zz.copy();modified[:,0]+=offset
    positive=zz[:,1:].argmax(1)+1
    margin=zz[np.arange(len(yy)),positive]-zz[:,0]
    # Before the smallest threshold, all samples predict their fixed best positive.
    # At each tied breakpoint, all tied samples switch to class0 together.
    values,group=np.unique(margin,return_inverse=True)
    correct=(positive==yy);increments=(yy==0).astype(int)-correct.astype(int)
    totals=int(correct.sum())+np.cumsum(np.bincount(group,weights=increments).astype(int))
    best=int(max(correct.sum(),totals.max()))
    # Independently enumerate one representative of every distinct prediction region.
    representatives=np.r_[np.nextafter(values[0],-np.inf),values]
    exhaustive=np.array([np.where(margin>b,positive,0)==yy for b in representatives]).sum(1)
    assert int(exhaustive.max())==best
    result=dict(status='completed',offset=offset,fit_items=int(fit.sum()),
        fit_objective_derivative=derivative(offset),
        original=describe(zz,yy),calibration_fit=describe(modified,yy),
        shared_offset_accuracy_oracle=dict(correct=best,items=len(yy),accuracy=best/len(yy),
            additional_correct=best-int((zz.argmax(1)==yy).sum()),
            irreducible_errors_under_this_constraint=len(yy)-best,
            unique_breakpoints=len(values),exhaustive_region_check=True,
            limits='Label-aware assessmentoracle within ONEsharedclass0offset only; not Macroceiling or independent evidence.'),
        model_inference=False,valid_read=False,test_read=False,cuda_used=False,L5='NOT_ACCEPTED')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
