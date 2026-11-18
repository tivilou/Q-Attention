"""Fixed positive-39 metrics and paired entity-group resampling."""
import hashlib
import numpy as np


def confusion(gold,pred,weights=None):
    gold=np.asarray(gold);pred=np.asarray(pred)
    assert gold.shape==pred.shape and gold.ndim==1 and gold.dtype.kind in 'iu' and pred.dtype.kind in 'iu'
    assert np.all((gold>=0)&(gold<40)) and np.all((pred>=0)&(pred<40))
    if weights is None:weights=np.ones(len(gold),dtype=np.int64)
    weights=np.asarray(weights);assert weights.shape==gold.shape and weights.dtype.kind in 'iu' and np.all(weights>=0)
    matrix=np.zeros((40,40),dtype=np.int64);np.add.at(matrix,(gold,pred),weights)
    return matrix


def from_confusion(matrix):
    matrix=np.asarray(matrix);assert matrix.shape==(40,40) and matrix.dtype.kind in 'iu' and np.all(matrix>=0)
    support=matrix.sum(1);predicted=matrix.sum(0);denominator=support+predicted
    f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(40,dtype=np.float64),where=denominator!=0)
    return dict(macro_f1=float(f1[1:].sum()/39),accuracy=float(np.trace(matrix)/matrix.sum()) if matrix.sum() else 0.,
        support=support.tolist(),predicted=predicted.tolist(),per_class_f1=f1.tolist(),confusion=matrix.tolist())


def paired_group_bootstrap(gold,predictions,groups,replicates=2000,seed=20261119):
    gold=np.asarray(gold);groups=list(groups);names=list(predictions)
    assert len(gold)==len(groups)>0 and 'native_quantum' in names and 'ecqp_quantum' in names
    assert gold.ndim==1 and gold.dtype.kind in 'iu' and np.all((gold>=0)&(gold<40))
    assert all(isinstance(g,str) and g for g in groups) and isinstance(replicates,int) and replicates>0
    unique=sorted(set(groups));mapping={g:i for i,g in enumerate(unique)}
    group_index=np.array([mapping[g] for g in groups],dtype=np.int64)
    per_condition={}
    for name,pred in predictions.items():
        pred=np.asarray(pred);assert pred.shape==gold.shape and pred.dtype.kind in 'iu' and np.all((pred>=0)&(pred<40))
        support=np.zeros((len(unique),40),dtype=np.int64);guessed=np.zeros_like(support);tp=np.zeros_like(support)
        np.add.at(support,(group_index,gold),1);np.add.at(guessed,(group_index,pred),1)
        correct=gold==pred;np.add.at(tp,(group_index[correct],gold[correct]),1)
        per_condition[name]=(tp,support+guessed)
    rng=np.random.default_rng(seed);draws={n:[] for n in names};draw_hash=hashlib.sha256()
    for _ in range(replicates):
        selected=rng.integers(0,len(unique),size=len(unique));counts=np.bincount(selected,minlength=len(unique))
        draw_hash.update(counts.astype('<i8').tobytes())
        for name,(tp,denominator) in per_condition.items():
            numerator=2*(counts@tp);denom=counts@denominator
            f1=np.divide(numerator,denom,out=np.zeros(40,dtype=np.float64),where=denom!=0)
            draws[name].append(float(f1[1:].sum()/39))
    result={}
    for name in names:
        delta=np.asarray(draws[name])-np.asarray(draws['native_quantum'])
        result[name]=dict(delta_vs_native_quantiles=np.quantile(delta,[.025,.975],method='linear').tolist())
    return dict(groups=len(unique),replicates=replicates,seed=seed,intervals=result,
        group_multiplicity_sha256=draw_hash.hexdigest(),
        scope='Conditional development group resampling; donor graph/coupling and training randomness not certified.')


def decision(metrics,primary_interval,fit_support):
    assert np.asarray(fit_support).shape==(40,) and np.all(np.asarray(fit_support)>=0)
    assert np.asarray(primary_interval).shape==(2,) and np.isfinite(primary_interval).all()
    assert primary_interval[0]<=primary_interval[1]
    assert all(np.isfinite(row['macro_f1']) and np.isfinite(row['ce']) for row in metrics.values())
    primary=metrics['ecqp_quantum']['macro_f1']-metrics['native_quantum']['macro_f1']
    ce=metrics['ecqp_quantum']['ce']-metrics['native_quantum']['ce']
    if primary<=-.05 and ce>=.1:return 'CATASTROPHIC_FIXED_SCREEN'
    lower,upper=primary_interval
    if lower<=0<=upper or any(x==0 for x in fit_support[1:]) or any(x==0 for x in metrics['ecqp_quantum']['support'][1:]):
        return 'INCONCLUSIVE_NO_PROMOTION'
    if lower>0 and primary>0 and all(metrics['ecqp_quantum']['macro_f1']>metrics[n]['macro_f1'] for n in ['mismatch_quantum','square_quantum']):
        return 'EXPLORATORY_SCREEN_POSITIVE'
    return 'NO_POSITIVE_SCREEN'
