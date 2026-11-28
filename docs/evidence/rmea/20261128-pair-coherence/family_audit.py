"""Source-bound cross-family macro audit. Report only within-protocol contrasts."""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json
import sys

INPUTS={
    'fock':('non-gaussian-optical-routing-20261125/full/completed-report/summary.json','4f4234a470b828b0d9a10a5da9a5a3ebaa78ad3ed4a591b92a0ed2eff8bf3d2b'),
    'number':('number-coherent-optical-routing-20261125/completed-report/summary.json','47f6f24092b259015ae93cf2fa24c9d1b3f43cfdcc2e2f19f422883e1eb8fe24'),
    'return':('relative-return-training-20261122/full-completed/summary.json','c1b86173e89daa17610cd87eed25623cf605ed5645305e98dd5f9d435cdff1ea'),
    'flow':('context-spectral-flow-20261125/summary.json','ca8d6f75cc8726580004a0c432ddee8d3dae36a51222b13a7c15da09b43a50c6'),
    'distinct':('distinct-token-evidence-20261125/summary.json','dc3ed63bc46f15fb193b456eb302fc68435f48469d701e81677e34938262c2c7'),
}


def metric(record):
    counts=record['confusion_matrix']
    assert len(counts)==40 and all(len(row)==40 and all(type(n) is int and n>=0 for n in row) for row in counts)
    double=[sum(counts[i])+sum(row[i] for row in counts) for i in range(40)]
    per=[F(2*counts[i][i],d) if d else F() for i,d in enumerate(double)]
    value=float(sum(per)/40)
    actual=record.get('macro_f1',record.get('all40_macro_f1'))
    assert abs(value-actual)<1e-12,(value,actual,record.keys())
    return dict(macro=value,positive39=float(sum(per[1:])/39),correct=sum(row[i] for i,row in enumerate(counts)),
                rows=sum(map(sum,counts)), classwise=per)


def eligible(q, dcontrol, classics, tables, rows, stage):
    qrow=metric(tables[q]);drow=metric(tables[dcontrol])
    crows={k:metric(tables[k]) for k in classics}
    assert all(v['rows']==rows for v in (qrow,drow,*crows.values()))
    key,best=max(crows.items(),key=lambda s:s[1]['macro'])
    return dict(stage=stage,candidate=q,dephased_control=dcontrol,strongest_classical=key,
                q=qrow['macro'],d=drow['macro'],classic=best['macro'],delta_dephased=qrow['macro']-drow['macro'],
                delta_classic=qrow['macro']-best['macro'],q_positive39=qrow['positive39'],
                classic_positive39=best['positive39'],delta_correct=qrow['correct']-best['correct'],
                stronger_class_count=sum(a>b for a,b in zip(qrow['classwise'],best['classwise'])),
                weaker_class_count=sum(a<b for a,b in zip(qrow['classwise'],best['classwise'])),
                recomputed=2+len(crows))


def run(root):
    all_results={}
    for kind,(name,digest) in INPUTS.items():
        raw=(root/'.codex-artifacts'/name).read_bytes()
        assert sha256(raw).hexdigest()==digest
        all_results[kind]=json.loads(raw)
    fs='formal single seed13, recorded TEST13418'
    ss='historical development screen, exposed TRAIN4096'
    f=all_results['fock']; n=all_results['number']; rr=all_results['return']
    assert f['qualification_gate'] is n['qualification_gate'] is rr['promotion_gate'] is False
    assert f['seed']==n['seed']==rr['seed']==13 and all(x['full_data_training'] for x in (f,n,rr))
    tables=lambda d:{k:v['test'] for key in ('results','reused_native_results') for k,v in d.get(key,{}).items()}
    n_tables=tables(n)
    f_tables=tables(f)
    assert f_tables['real12']['confusion_matrix']==n_tables['real12']['confusion_matrix']
    profiles={}
    profiles['gen73']=eligible('fock1','parity_fock1',('gaussian3_born','rbf12','softmax12','real12'),f_tables,13418,fs)
    profiles['gen84']=eligible('number_coherent02','number_mixed02',
        ('curvature_matched_gaussian02','real12','softmax12'),n_tables,13418,fs)
    profiles['relative_return']=eligible('return_coherent','return_dephased',('native_square',),tables(rr),13418,fs)
    profiles['relative_return']['native_reference']=profiles['relative_return'].pop('strongest_classical')
    profiles['relative_return']['delta_native_reference']=profiles['relative_return'].pop('delta_classic')
    profiles['relative_return']['comparison_limit']='native_square is a same-protocol native routing reference, not an independently trained classical replacement; exclude from fair-classical comparison.'
    for id, q, dctrl, classic in [('flow','quantum_flow','dephased',('real12_flow','gaussian_base')),
                                  ('distinct','quantum_distinct','dephased_distinct',('gaussian_distinct','earlier/native_gaussian_covariance'))]:
        ex=all_results[id]
        assert ex['test_read'] is ex['valid_read'] is False and ex['assessment_items']==4096 and ex['train_items']==8192
        ex_table=ex['results']|ex['reused_results']
        profiles[id]=eligible(q,dctrl,classic,ex_table,4096,ss)
    profiles['flow']['comparison_limit']='Base dephasing is a held protocol reference. This is not a same-spectral-flow ablation.'
    assert all(x['delta_classic']<0 for k,x in profiles.items() if k!='relative_return')
    assert abs(profiles['gen73']['delta_classic']+.00416278566)<1e-8
    return dict(status='PASS_HISTORICAL_FAMILY_CONTRASTS_NO_POOLED_EFFECT',profiles=profiles,
                exact_input_sha256={k:v[1] for k,v in INPUTS.items()},
                distinct_inputs=5,scope='Historical aggregate confusion matrices. No task files, output pairing, reweighting or protocol change.',
                new_task_forwards=0,optimizer_updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False))