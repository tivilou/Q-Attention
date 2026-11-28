"""Reconcile original and count-matched seed-13 effects without pooling protocols."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(sys.argv[1])
def loaded(relative,digest):
    raw=(ROOT/relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==digest
    return json.loads(raw)

def f1(cm):
    n=len(cm)
    assert n==40 and all(len(row)==40 for row in cm)
    den=[sum(cm[i])+sum(row[i] for row in cm) for i in range(n)]
    return sum(F(2*cm[i][i],den[i]) if den[i] else F(0) for i in range(n))/n

def run():
    original=loaded('reports/q_rmea/20261027T075438Z_seed13/summary.json',
                    '0d1c47938228397df39b792380f73b4b3baeb047ff772f58ccfb70a3ddda4e1e')
    current=loaded('docs/evidence/rmea/20261127-positive-anchor/math-result.json',
                   '3041a5d79b8a44325ff92834e384f97fda53e9c11928ef80650ab6db5418fe9c')
    assert original['seed']==13 and original['full_data_training'] and original['common_initialization_equal']
    m=original['results']
    for mode in m.values():
        cm=mode['test']['confusion_matrix']
        assert sum(map(sum,cm))==13418 and mode['updates']==9140
        assert abs(float(f1(cm))-mode['test']['macro_f1'])<1e-12
        assert mode['selected_epoch']==max(range(1,21),
            key=lambda i:mode['valid_history'][i-1]['macro_f1'])
    a=m['quantum']['test']['macro_f1']
    peers=('relation_softmax','real_overlap')
    best=max(peers,key=lambda k:m[k]['test']['macro_f1'])
    q_vs_d=a-m['dephased']['test']['macro_f1']
    q_vs_classical=a-m[best]['test']['macro_f1']
    b=current['gen49']['macro']; assert current['gen49']['passed_required']==1
    assert q_vs_d>0 and q_vs_classical>0 and b['quantum']<min(b[k] for k in ('dephased','real12','softmax12'))
    p=current['gen98'];assert p['passed_required']==10 and p['full_data']=='NOT_STARTED'
    return dict(status='PASS_PROTOCOL_SEPARATED_SEED13_INITIAL_FRONTIER',historical={
        'delta_vs_strongest_declared_classical':q_vs_classical,'delta_vs_dephased':q_vs_d,
        'classic_best':best,'q_registered_count':m['quantum']['parameter_count'],
        'classic_registered_count':m[best]['parameter_count'],'original_CM_recomputed':5},
        matched_full={'delta_vs_d':b['quantum']-b['dephased'],
        'delta_vs_strongest_declared_classical':b['quantum']-max(b[k] for k in ('real12','softmax12'))},
        gen98_original_classical_gate_margin=p['q_delta']['reused/earlier/native_gaussian_covariance'],
        scope='Independent protocols. Selection rechecked in original series only. Difference of contrasts does not identify a capacity, cache, or mechanism effect.',
        new_runs=0,updates=0,gpu_allocations=0)

if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))