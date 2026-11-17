"""Frozen native-float32 descriptive counts; no threshold selection."""
import numpy as np


def count_batch(scores,gold):
    z=np.asarray(scores);y=np.asarray(gold)
    assert z.dtype==np.float32 and z.ndim==2 and z.shape[1]==40
    assert y.shape==(len(z),) and y.dtype.kind in 'iu'
    assert np.isfinite(z).all() and (y>=0).all() and (y<40).all()
    prediction=z.argmax(1)
    confusion=np.zeros((40,40),np.int64)
    np.add.at(confusion,(y,prediction),1)
    eligible=(y>0)&(prediction==0)
    subset=z[eligible];target=y[eligible]
    row=np.arange(len(subset))
    positive=subset[:,1:]
    positive_max=positive.max(1) if len(subset) else np.empty(0,np.float32)
    top_ties=(positive==positive_max[:,None]).sum(1)>1
    # Explicitly exclude the gold column before comparing the other38 positives.
    wrong=positive.copy()
    if len(subset):wrong[row,target-1]=-np.inf
    winner=(subset[row,target]>wrong.max(1)) if len(subset) else np.empty(0,bool)
    null_ties=(subset[:,0]==positive_max) if len(subset) else np.empty(0,bool)
    return dict(total=len(z),gold_positive=int((y>0).sum()),predicted_null=int((prediction==0).sum()),
        eligible=int(eligible.sum()),strict_gold_winner=int(winner.sum()),
        nonnull_maximum_ties=int(top_ties.sum()),null_positive_maximum_ties=int(null_ties.sum()),
        confusion=confusion.tolist())


def finish(counts):
    result=dict(counts)
    n,k=result['eligible'],result['strict_gold_winner']
    assert 0<=k<=n and n<=result['total']
    result['proportion']=k/n if n else None
    result['criterion']='UNDEFINED_EMPTY' if n==0 else ('PASS_DEFINED' if 2*k>n else 'FAIL_DEFINED')
    return result
