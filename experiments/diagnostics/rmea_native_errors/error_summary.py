"""Read existing frozen TRAIN predictions; no inference, fitting or CUDA use."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT = Path('/hy-tmp/Q-Attention')
REFERENCE = ROOT/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz'
EXPECTED = 'ff920146542e7af85f90df5ececfa62f0742be4e90f0db4de8645946e043d833'


def metric(z, y):
    n = len(y); order = np.argsort(-z, axis=1, kind='stable')
    pred = order[:, 0]; ranks = (order == y[:, None]).argmax(1)+1
    cm = np.bincount(y*40+pred, minlength=1600).reshape(40, 40)
    den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*cm.diagonal(), den, out=np.zeros(40), where=den>0)
    shifted = z-z.max(1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(1))
    ce = logsum-shifted[np.arange(n), y]
    confidence = np.exp(-logsum)
    wrong = pred != y
    tp = int(cm[1:, 1:].diagonal().sum())
    fp = int(cm[:, 1:].sum()-tp); fn = int(cm[1:, :].sum()-tp)
    rows = []
    for label in range(40):
        rows.append(dict(label=label, support=int(cm[label].sum()),
            predicted=int(cm[:, label].sum()), true_positive=int(cm[label,label]),
            f1=float(f1[label]), errors=int(cm[label].sum()-cm[label,label])))
    categories = dict(positive_to_zero=int(((y>0)&(pred==0)).sum()),
        zero_to_positive=int(((y==0)&(pred>0)).sum()),
        wrong_positive_relation=int(((y>0)&(pred>0)&wrong).sum()))
    assert sum(categories.values()) == int(wrong.sum())
    q = lambda a: np.quantile(a, [0,.25,.5,.75,1]).tolist() if len(a) else None
    return dict(items=n, errors=int(wrong.sum()), accuracy=float((~wrong).mean()),
        ce=float(ce.mean()), positive39_macro_f1=float(f1[1:].mean()),
        positive_micro_f1=2*tp/(2*tp+fp+fn), zero_fraction=float((y==0).mean()),
        errors_by_type=categories,
        rank_counts={str(k):int((ranks==k).sum()) for k in range(1,41)},
        fixed_top_k_accuracy_oracle={str(k):dict(accuracy=float((ranks<=k).mean()),
            additional_correct=int(((ranks<=k)&wrong).sum()),
            headroom_accuracy_points=float(((ranks<=k)&wrong).sum()/n)) for k in (2,3,5)},
        confidence_quantiles_correct=q(confidence[~wrong]),
        confidence_quantiles_wrong=q(confidence[wrong]),
        ce_sum_by_error_type=dict(correct=float(ce[~wrong].sum()),
            positive_to_zero=float(ce[(y>0)&(pred==0)].sum()),
            zero_to_positive=float(ce[(y==0)&(pred>0)].sum()),
            wrong_positive_relation=float(ce[(y>0)&(pred>0)&wrong].sum())),
        classes=rows, confusion=cm.tolist()), pred


def main():
    started = time.time()
    digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest(); assert digest == EXPECTED
    with np.load(REFERENCE, allow_pickle=False) as a:
        z=a['logits']; y=a['labels']; calibration=a['calibration']
        lengths=a['lengths']; groups=a['groups']
        assert z.shape==(8192,40) and y.shape==(8192,) and np.isfinite(z).all()
        assert y.dtype.kind in 'iu' and y.min()>=0 and y.max()<40
        assert calibration.sum()==4192 and (~calibration).sum()==4000
        assert not set(groups[calibration]) & set(groups[~calibration])
        parts = {}
        for name, select in [('all',np.ones(8192,dtype=bool)),
                             ('calibration',calibration),('assessment',~calibration)]:
            row,pred=metric(z[select],y[select]); parts[name]=row
            wrong=pred!=y[select]; sizes=lengths[select]
            row['length_bins']=[dict(lower=lo,upper=hi,items=int(((sizes>=lo)&(sizes<=hi)).sum()),
                errors=int((wrong&(sizes>=lo)&(sizes<=hi)).sum()))
                for lo,hi in ((1,16),(17,32),(33,64),(65,128),(129,1000000))]
            assert sum(q['items'] for q in row['length_bins'])==row['items']
    result=dict(status='completed', seconds=time.time()-started,
        reference_sha256=digest, checkpoint_sha256='2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51',
        scope='Previously exposed first8192TRAIN; source checkpoint may have trained on all rows. Descriptive planning, not independent generalization or causal attribution.',
        fixed_pool_oracle=dict(status='exact_under_stated_assumptions',metric='accuracy',
            constraint='Choose exactly one class from frozen native topK; stable class-ID tie ordering.',
            interpretation='Label-aware empirical upper bound attained by an oracle chooser, not a trained predictor. TopK headroom is not headroom for unrestricted models or Macro-F1.'),
        unrestricted_macro_ceiling=dict(status='unresolved',
            reason='No identifiable optimal classifier or input-ambiguity bound; nominal1 is not an attainable-performance certificate.'),
        model_inference=False, optimizer_updates=0, cuda_used=False,
        valid_read=False,test_read=False,cos_writes=0,partitions=parts,L5='NOT_ACCEPTED')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
