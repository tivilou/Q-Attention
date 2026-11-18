"""Recompute 39-positive Macro-F1 from all published confusion matrices."""
import json,math
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent
    x=json.loads((here/'filter-results.json').read_bytes());rows=x['cells']
    assert [(r['seed'],r['head']) for r in rows]==[(s,h) for s in [29,53,71,97,113] for h in ['quantum','real_overlap']]
    for row in rows:
        assert row['original_confusion_exact'] and row['identity_argmax_exact']
        for name,metric in row['evaluation'].items():
            cm=metric['confusion'];assert len(cm)==40 and all(len(r)==40 for r in cm)
            assert all(type(v)==int and v>=0 for r in cm for v in r)
            assert sum(map(sum,cm))==metric['items']
            f=[]
            for i in range(1,40):
                denominator=sum(cm[i])+sum(r[i] for r in cm)
                f.append(2*cm[i][i]/denominator if denominator else 0)
            # Python sequential sum and NumPy pairwise mean round differently.
            assert math.isclose(sum(f)/39,metric['macro_f1'],rel_tol=0,abs_tol=1e-15)
            assert math.isclose(sum(cm[i][i] for i in range(40))/metric['items'],metric['accuracy'],rel_tol=0,abs_tol=1e-16)
            if name!='baseline':assert row['macro_deltas'][name]==metric['macro_f1']-row['evaluation']['baseline']['macro_f1']
        assert row['evaluation']['baseline']['confusion']==row['evaluation']['identity']['confusion']
        m=row['evaluation'];wins=m['protected']['macro_f1']>max(m[k]['macro_f1'] for k in ['baseline','unprotected','score_quadratic'])
        assert row['protected_beats_both']==wins
        assert sum(row['routing']['reasons'].values())==19584
        assert row['score_probe_fit']['penalty']==.01 and row['score_probe_fit']['input_features']==80
        assert row['score_probe_fit']['gradient_infinity_norm']<=1e-8
    criterion='ALL5_PROTECTED_BEATS_CONTROLS' if all(r['protected_beats_both'] for r in rows if r['head']=='quantum') else 'FIXED_INTERVENTION_CRITERION_NOT_MET'
    assert x['criterion']==criterion and x['L5']=='NOT_ACCEPTED'
    assert not x['head_training'] and x['probe_fit'] and not x['original_test_evaluation'] and x['cos_requests']==0
    print(json.dumps(dict(status='PASS_COMPACT_FILTER_RECOMPUTATION',cells=10,confusion_matrices=50,criterion=criterion,L5='NOT_ACCEPTED')))
if __name__=='__main__':main()
