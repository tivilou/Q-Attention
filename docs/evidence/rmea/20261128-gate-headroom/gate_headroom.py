"""Exact post-hoc ceilings with fixed positive predictions and frozen evaluation rows."""

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def confusion(gold, predicted, classes):
    return np.bincount(classes*gold+predicted, minlength=classes**2).reshape(classes, classes)


def macro(cm):
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.divide(2*cm.diagonal(), den, out=np.zeros(len(cm), dtype=float), where=den > 0).mean())


def soft_max(a, b):
    return np.maximum(a, b)


def unrestricted_oracle(gold, positive, classes):
    """Maximize all-class macro-F1 over {0, fixed positive pick} per row."""
    supports = np.bincount(gold, minlength=classes)
    match = (gold == positive)
    correct = np.bincount(gold[match], minlength=classes)
    wrong = (gold > 0) & ~match
    capacities = np.bincount(positive[wrong], minlength=classes)
    assert gold.min() == 0 and positive.min() > 0
    # Correctly rejecting true nulls and accepting matching positive rows
    # weakly improves all affected class scores. On other positive rows,
    # fixing total K accepted makes the null score independent of how K is
    # assigned. A convex box objective has an optimum at a vertex; each
    # positive class therefore absorbs either all or none of its wrong picks.
    den = supports+correct
    numerator = 2*correct
    best = np.array([0.])
    for j in range(1, classes):
        n, a, b = capacities[j], numerator[j], den[j]
        reject = a/b if b else 0.0
        accept = a/(b+n) if b+n else 0.0
        new = np.full(len(best)+n, -np.inf)
        new[:len(best)] = best+reject
        new[n:] = soft_max(new[n:], best+accept)
        best = new
    wrong_count = int(wrong.sum())
    all_scores = (best + 2*supports[0]/(2*supports[0]+wrong_count-np.arange(len(best))))/classes
    optimum = int(all_scores.argmax())
    return dict(exact_macro=float(all_scores[optimum]), wrong_positive_rows=wrong_count,
                eligible_correct_positive_rows=int(match.sum()), wrong_positive_rows_accepted=optimum,
                oracle_all_wrong_reject_macro=float(all_scores[0]))


def null_repair_curve(cm):
    """Best possible macro after exactly k true-null false-positive reversals."""
    n = int(cm[0, 1:].sum())
    baseline = cm[0, 0]
    den = cm.sum(axis=0)+cm.sum(axis=1)
    best = np.array([0.])
    for j in range(1, len(cm)):
        cap = int(cm[0, j])
        new = np.full(len(best)+cap, -np.inf)
        for repaired in range(cap+1):
            score = 2*cm[j, j]/(den[j]-repaired) if den[j]>repaired else 0.0
            view = new[repaired:repaired+len(best)]
            view[:] = soft_max(view, best+score)
        best = new
    repaired = np.arange(n+1)
    return (best+2*(baseline+repaired)/(den[0]+repaired))/len(cm)


def positive_wrong_rank_repair(gold, pred, fixed, classes):
    allowed = (gold > 0) & (pred == 0) & (fixed == gold)
    return macro(confusion(gold, np.where(allowed, fixed, pred), classes)), int(allowed.sum())


def tiny_exact_fixture():
    rng = np.random.default_rng(4827)
    for _ in range(64):
        gold = np.concatenate((np.arange(4), rng.integers(0, 4, 5)))
        best_positive = rng.integers(1, 4, len(gold))
        base = np.where(rng.integers(0, 2, len(gold)), best_positive, 0)
        exact = max(macro(confusion(gold, np.array(mask)*best_positive, 4))
                    for mask in itertools.product((0, 1), repeat=len(gold)))
        assert abs(exact-unrestricted_oracle(gold,best_positive,4)['exact_macro']) < 1e-12
        curve = null_repair_curve(confusion(gold,base,4))
        rows = np.flatnonzero((gold == 0) & (base != 0))
        assert len(curve) == len(rows)+1
        for size, target in enumerate(curve):
            values = []
            for repaired in itertools.combinations(rows,size):
                pred = base.copy()
                pred[list(repaired)] = 0
                values.append(macro(confusion(gold,pred,4)))
            assert abs(max(values)-target) < 1e-12
    return 64


def file_path(pin):
    path = Path(pin['path'])
    assert path.is_absolute() and path.is_file() and not path.is_symlink()
    assert path.stat().st_size <= 64*1024**2
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for segment in iter(lambda: stream.read(1024*1024), b''):
            digest.update(segment)
    assert digest.hexdigest() == pin['sha256']
    return path


def audit(cfg):
    small = tiny_exact_fixture()
    pins = cfg['pins']
    path = {k: file_path(v) for k, v in pins.items()}
    cohort = json.loads(path['cohort'].read_bytes())
    indices = np.asarray(cohort['assessment'], dtype=np.int64)
    full_gold = np.load(path['train_targets'], allow_pickle=False)
    assert full_gold.shape == (58465,) and full_gold.min() == 0 and full_gold.max() == 39
    assert indices.shape == (4096,) and len(set(indices)) == 4096
    fit = cohort['fit']
    assert len(fit) == 8192 and not set(indices).intersection(fit)
    gold = full_gold[indices]
    assert cohort['target_sha256'] == pins['train_targets']['sha256'] and (gold>0).sum() == 1365
    records, curves = {}, {}
    for collection, variants in [('distinct', ('Q', 'D', 'G')), ('reference', ('Gcov',))]:
        with np.load(path[collection], allow_pickle=False) as existing:
            assert np.array_equal(existing['assessment_indices'], indices)
            assert np.array_equal(existing['labels'], full_gold)
            assert np.array_equal(np.flatnonzero(existing['calibration']), np.sort(fit))
            for name in variants:
                logits = existing[cfg['keys'][name]+'_logits']
                assert logits.shape == (4096, 40) and np.isfinite(logits).all()
                pred = logits.argmax(axis=1)
                pos = logits[:, 1:].argmax(axis=1)+1
                assert np.array_equal(np.where(pred>0,pos,0), pred)
                cm = confusion(gold, pred, 40)
                assert cm.tolist() == cfg['confusions'][name] and np.all(cm.sum(axis=1) > 0)
                curve = null_repair_curve(cm)
                assert abs(curve[0] - macro(cm)) < 1e-12
                assert np.all(np.diff(curve)>=-1e-12)
                x = unrestricted_oracle(gold,pos,40)
                assert x['eligible_correct_positive_rows'] == cfg['historic_positive_rank_1'][name]
                v, c = positive_wrong_rank_repair(gold,pred,pos,40)
                assert min(x['exact_macro']-curve[-1], x['exact_macro']-v) >= -1e-12
                records[name] = dict(observed=macro(cm), fixed_positive_top1_oracle=x,
                                     false_positive_only_ceiling=float(curve[-1]), false_positives=int(cm[0,1:].sum()),
                                     positive_only_ceiling=v, isolated_correct_positive_misses=c)
                curves[name] = curve
    strongest = max(records, key=lambda k:records[k]['observed'])
    assert strongest == 'Gcov'
    target = records[strongest]['observed']
    q = curves['Q']

    def count(v):
        eligible = np.flatnonzero(q>=v)
        return int(eligible[0]) if len(eligible) else None

    return dict(status='PASS_EXACT_FIXED_POSITIVE_RANK_MACRO_HEADROOM', potential_kind='outcome-aware empirical oracle; not deployable',
                baseline=strongest, rows=len(gold), zero_label_count=int((gold==0).sum()), fixed_classes=40,
                exposed_population=True, pad_or_threshold_selected=False,
                source_ids={k: v['sha256'] for k, v in pins.items()}, exhaustive_tiny_cases=small, source_label_hash=cfg['public_label_file_sha256'],
                arms=records, min_fp_repairs_to_match_Gcov=count(target), min_fp_repairs_to_exceed_Gcov_by_one_pp=count(target+0.01),
                max_Q_under_observed_exact_positive_picks_delta_vs_gcov=records['Q']['fixed_positive_top1_oracle']['exact_macro']-target,
                task_forwards=0, optimizer_updates=0, gpu_allocations=0, new_labels_read=0, official_splits_read=0, rows_exported=0)


if __name__ == '__main__':
    print(json.dumps(audit(json.load(sys.stdin)), allow_nan=False))