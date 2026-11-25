"""Read existing gen98 predictions and cached TRAIN roles; emit aggregates only.

No model import/forward, no h-feature load, no dataset VALID/TEST access or writes.
The feature list and FIT-median cuts are fixed in strata-design.json beforehand.
"""
from pathlib import Path
import hashlib, json, os, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

started = time.monotonic()
root = Path('/hy-tmp/Q-Attention')
out = root / 'control/distinct-token-evidence-20261125/output'
cache = root / 'inputs/native-baseline/discovery/cache'
models = ('quantum_distinct', 'dephased_distinct', 'gaussian_distinct')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(cache.parent / 'manifest.json') == 'dfb347aef35657c5463486e1043aca0617ccc11d3aca1eea851a59af15ec5f49'
assert sha(out / 'diagnostic.private.npz') == '3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307'
cohort_path = root / 'control/staged-bottleneck-design-20261125/cohort.private.json'
assert sha(cohort_path) == '8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0'
cohort = json.loads(cohort_path.read_bytes())
manifest = json.loads((cache.parent / 'manifest.json').read_bytes())
for name in ('train-offsets.npy', 'train-roles.npy', 'train-targets.npy'):
    assert sha(cache / name) == manifest['cache'][name]['sha256']
with np.load(out / 'diagnostic.private.npz', allow_pickle=False) as diagnostic:
    assessment = diagnostic['assessment_indices'].copy()
    fit = np.flatnonzero(diagnostic['calibration'])
    assert len(fit) == 8192 and len(assessment) == 4096
    assert set(map(int, fit)) == set(cohort['fit'])
    assert assessment.tolist() == cohort['assessment']
    targets = diagnostic['labels'].copy()
    assert np.array_equal(targets, np.load(cache / 'train-targets.npy', allow_pickle=False))
    assert not set(cohort['selected_groups'][str(i)] for i in fit).intersection(
        cohort['selected_groups'][str(i)] for i in assessment)
    prediction = {m: diagnostic[m + '_logits'].argmax(-1) for m in models}
offsets = np.load(cache / 'train-offsets.npy', allow_pickle=False)
roles = np.load(cache / 'train-roles.npy', mmap_mode='r', allow_pickle=False)


def features(indices):
    lengths, gaps, order = [], [], []
    for index in indices:
        lo, hi = map(int, offsets[index:index + 2])
        role = roles[lo:hi]
        s = np.flatnonzero(role[:, 2] == 1)
        o = np.flatnonzero(role[:, 3] == 1)
        assert len(s) and len(o)
        assert np.array_equal(s, np.arange(s[0], s[-1] + 1))
        assert np.array_equal(o, np.arange(o[0], o[-1] + 1))
        lengths.append(hi - lo)
        gaps.append(max(int(o[0] - s[-1] - 1), int(s[0] - o[-1] - 1), 0))
        overlap = max(s[0], o[0]) <= min(s[-1], o[-1])
        order.append('overlap' if overlap else ('subject_first' if s[0] < o[0] else 'object_first'))
    return np.asarray(lengths), np.asarray(gaps), np.asarray(order)


fit_length, fit_gap, _ = features(fit)
length, gap, order = features(assessment)
cuts = dict(length_fit_median=float(np.median(fit_length)), gap_fit_median=float(np.median(fit_gap)))
short = length <= cuts['length_fit_median']
near = gap <= cuts['gap_fit_median']
gold = targets[assessment]
assert np.array_equal(np.unique(gold), np.arange(40))
global_support = np.bincount(gold, minlength=40)
correct = {m: prediction[m] == gold for m in models}
pair_code = (4 * correct[models[0]] + 2 * correct[models[2]] + correct[models[1]]).astype(np.int64)
global_summary = json.loads((out / 'summary.json').read_bytes())


def metrics(cm):
    tp = np.diag(cm)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(40, dtype=float), where=denominator > 0)
    return dict(all40_macro_f1=float(f1.mean()), correct=int(tp.sum()), per_class_f1=f1.tolist())


def aggregate(name, selection):
    y = gold[selection]
    support = np.bincount(y, minlength=40)
    cm = {m: np.bincount(40 * y + prediction[m][selection], minlength=1600).reshape(40, 40) for m in models}
    by_class = np.bincount(8 * y + pair_code[selection], minlength=320).reshape(40, 8)
    assert np.array_equal(by_class.sum(1), support)
    class_net = np.diag(cm[models[0]]) - np.diag(cm[models[2]])
    contribution = np.mean(class_net / global_support)
    paired = by_class.sum(0)
    q_only_g = sum(int(paired[i]) for i in range(8) if i & 4 and not i & 2)
    g_only_q = sum(int(paired[i]) for i in range(8) if i & 2 and not i & 4)
    return dict(name=name, rows=int(selection.sum()), support=support.tolist(),
        paired_correct_q_g_d_codes=paired.tolist(), paired_by_class=by_class.tolist(),
        q_correct_g_wrong=q_only_g, g_correct_q_wrong=g_only_q,
        confusion_matrices={m:v.tolist() for m,v in cm.items()},
        metrics={m:metrics(v) for m,v in cm.items()},
        additive_global_balanced_accuracy_q_minus_g=float(contribution))


partitions = dict(
    length=[('length_le_fit_median', short), ('length_gt_fit_median', ~short)],
    gap=[('gap_le_fit_median', near), ('gap_gt_fit_median', ~near)],
    role_order=[(x, order == x) for x in ('subject_first', 'object_first', 'overlap')],
    length_gap=[(f'length_{a}_gap_{b}', (short == (a == 'le')) & (near == (b == 'le')))
                for a in ('le', 'gt') for b in ('le', 'gt')])
all_rows = aggregate('all', np.ones(len(gold), dtype=bool))
for m in models:
    assert all_rows['confusion_matrices'][m] == global_summary['results'][m]['confusion_matrix']
    assert abs(all_rows['metrics'][m]['all40_macro_f1'] - global_summary['results'][m]['all40_macro_f1']) < 1e-12
tables = {k: [aggregate(name, selection) for name, selection in bins] for k, bins in partitions.items()}
for rows in tables.values():
    assert sum(row['rows'] for row in rows) == len(gold)
    assert np.array_equal(sum(np.asarray(row['support']) for row in rows), global_support)
    assert np.array_equal(sum(np.asarray(row['paired_by_class']) for row in rows), np.asarray(all_rows['paired_by_class']))
    for m in models:
        assert np.array_equal(sum(np.asarray(row['confusion_matrices'][m]) for row in rows), np.asarray(all_rows['confusion_matrices'][m]))
    assert abs(sum(row['additive_global_balanced_accuracy_q_minus_g'] for row in rows) - all_rows['additive_global_balanced_accuracy_q_minus_g']) < 1e-12
result = dict(status='PASS_EXISTING_PREDICTIONS_EXPLORATORY_STRATA', source_summary_sha256=sha(out / 'summary.json'),
    source_diagnostic_sha256=sha(out / 'diagnostic.private.npz'), cohort_sha256=sha(cohort_path),
    source_role_sha256=sha(cache / 'train-roles.npy'), source_offsets_sha256=sha(cache / 'train-offsets.npy'),
    threshold_source='8192 FIT rows; exact numpy.median; no outcome threshold search', cuts=cuts,
    paired_code_definition='4*Q_correct + 2*G_correct + D_correct', overall=all_rows, partitions=tables,
    seconds=time.monotonic() - started, raw_arrays_exported=False, new_model_forwards=0, updates=0,
    dataset_valid_test_read=False, gpu_allocations=0, cos_writes=0,
    scope='Exposed development data; all40 fixed Macro denominator incl zero support. Class mixtures differ by stratum. Additive balanced-accuracy contribution is descriptive and not the primary metric or a candidate gate. No causal or quantum-specific inference.')
print(json.dumps(result, allow_nan=False))
