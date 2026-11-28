"""Min/max recoverable macro for any fixed count of true-null repairs in an audited CM."""

from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np

from gate_headroom import confusion, macro, null_repair_curve


def minimum_curve(cm):
    base = cm[0, 0]
    den = cm.sum(axis=0)+cm.sum(axis=1)
    worst = np.array([0.])
    for j in range(1, len(cm)):
        cap = int(cm[0, j])
        next_worst = np.full(worst.size+cap, np.inf)
        for repaired in range(cap+1):
            f1 = 2*cm[j, j]/(den[j]-repaired) if den[j]>repaired else 0.0
            view = next_worst[repaired:repaired+worst.size]
            view[:] = np.minimum(view, worst+f1)
        worst = next_worst
    count = np.arange(worst.size)
    return (worst + 2*(base+count)/(den[0]+count))/len(cm)


def self_test():
    rng = np.random.default_rng(704)
    for _ in range(96):
        y = np.concatenate((np.zeros(4, dtype=int), np.arange(4), rng.integers(0, 4, 2)))
        z = rng.integers(0, 4, len(y))
        null = np.flatnonzero((y==0)&(z>0))
        lower = minimum_curve(confusion(y,z,4))
        upper = null_repair_curve(confusion(y,z,4))
        for k in range(len(null)+1):
            scores = []
            for group in itertools.combinations(null,k):
                v = z.copy()
                v[list(group)] = 0
                scores.append(macro(confusion(y,v,4)))
            assert abs(lower[k]-min(scores)) < 1e-12 and abs(upper[k]-max(scores)) < 1e-12


def run(summary_path, rank_path, headroom_path):
    summary_path = Path(summary_path)
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert digest == 'dc3ed63bc46f15fb193b456eb302fc68435f48469d701e81677e34938262c2c7'
    summary, rank, result = [json.loads(Path(p).read_bytes()) for p in (summary_path,rank_path,headroom_path)]
    cm = np.asarray(summary['results']['quantum_distinct']['confusion_matrix'])
    ref = np.asarray(summary['reused_results']['earlier/native_gaussian_covariance']['confusion_matrix'])
    q = result['arms']['Q']
    assert cm.shape == ref.shape == (40,40) and cm.sum() == ref.sum() == result['rows'] == 4096
    assert np.all(cm.sum(1) == ref.sum(1)) and cm[0,1:].sum()==rank['reports']['Q']['null_false_positive']==391
    assert abs(macro(cm)-q['observed']) < 1e-12 and abs(macro(ref)-result['arms']['Gcov']['observed']) < 1e-12
    lower, upper = minimum_curve(cm), null_repair_curve(cm)
    assert lower.shape == upper.shape == (392,) and np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(np.diff(lower) >= -1e-12) and np.all(np.diff(upper) >= -1e-12)
    assert np.all(lower<=upper+1e-12) and abs(lower[-1]-upper[-1]) < 1e-12
    assert abs(lower[0]-macro(cm)) < 1e-12 and abs(upper[-1]-q['false_positive_only_ceiling']) < 1e-12
    self_test()

    def first(values, target):
        eligible = np.flatnonzero(values>=target)
        return int(eligible[0]) if eligible.size else None

    pair = {}
    for name, add in [('match_Gcov', 0.0), ('one_pp_above_Gcov', 0.01)]:
        target = macro(ref)+add
        pair[name] = dict(target=target, best_case_min_count=first(upper,target),
                          any_allocations_min_count=first(lower,target))
        assert pair[name]['best_case_min_count']==result['min_fp_repairs_to_match_Gcov' if not add else
                                                        'min_fp_repairs_to_exceed_Gcov_by_one_pp']
    return dict(status='PASS_EXACT_COUNT_BOUNDS_FROM_EXISTING_CM', summary_sha256=digest, bounds=pair,
            at_55_repairs=dict(lower=float(lower[55]), upper=float(upper[55])),
            oracle_at_15_repairs=dict(lower=float(lower[15]), upper=float(upper[15])),
            positive_row_decisions_changed=False, local_exact_tiny_cases=96,
            exceptions='Only original true-null false positives may be reverted, with all positive row decisions fixed and no new errors.',
            new_task_forwards=0, gpu_allocations=0, cell_or_prediction_rows_exported=0)


if __name__=='__main__':
    print(json.dumps(run(*sys.argv[1:]),indent=2,allow_nan=False))