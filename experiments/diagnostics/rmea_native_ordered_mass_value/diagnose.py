"""Frozen TRAIN diagnostic of ordered native attention mass/value records."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import itertools
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    pending = p.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(p)


def centered(x):
    return x - x.mean(-1, keepdims=True)


def records(q, u, roles):
    """Inputs [relation, validtoken], [validtoken,4]; no padded records."""
    n = q.shape[1]
    assert q.shape == u.shape and roles.shape == (n, 4) and n > 0
    m = q / q.mean(-1, keepdims=True)
    s = (m * u).mean(-1, keepdims=True)
    c1, c2 = m - 1, m * (u - s)
    prefix1, prefix2 = np.cumsum(c1, axis=-1) - c1, np.cumsum(c2, axis=-1) - c2
    area = (prefix1 * c2 - prefix2 * c1).sum(-1) / n**2
    pos = np.arange(1, n + 1) * 2 - n - 1
    trend = (pos * c2).sum(-1) / n**2
    f = roles - roles.mean(0, keepdims=True)
    entity = np.concatenate([c2 @ f / n, c1 @ f / n], axis=-1)
    return area, trend, entity, s[:, 0], np.stack([c1, c2], -1)


def residualize(feature, logits, trend, entity, lengths, calibration):
    """Per-relation nuisance OLS; fit and standardization use calibration only."""
    out = np.empty_like(feature); fits = []; ranks = []
    for r in range(40):
        x = np.column_stack([centered(logits), trend[:, r], entity[:, r], np.log(lengths)])
        mu, sd = x[calibration].mean(0), x[calibration].std(0)
        sd = np.where(sd > 1e-12, sd, 1.)
        x = np.column_stack([np.ones(len(x)), (x - mu) / sd])
        coef, _, rank, _ = np.linalg.lstsq(x[calibration], feature[calibration, r], rcond=1e-10)
        out[:, r] = feature[:, r] - x @ coef
        fits.append(np.concatenate([mu, sd, coef])); ranks.append(int(rank))
    return out, np.array(fits), ranks


def unit_rms(x, epsilon=1e-12):
    x = centered(x)
    rms = np.sqrt((x**2).mean(-1))
    return np.divide(x, rms[:, None], out=np.zeros_like(x), where=rms[:, None] > epsilon), rms


def bootstrap(values, group_ids, count, seed):
    """Resample exact-sentence groups; ratio estimator keeps all sampled rows."""
    _, inverse = np.unique(group_ids, return_inverse=True)
    groups = int(inverse.max()) + 1
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=values[:, j], minlength=groups)
                     for j in range(values.shape[1])], -1)
    rng = np.random.default_rng(seed); estimates = []
    for _ in range(0, count, 100):
        sample = rng.integers(0, groups, size=(min(100, count-len(estimates)), groups))
        z = sums[sample].sum(1) / counts[sample].sum(1)[:, None]
        estimates.extend(z)
    return np.asarray(estimates)


def fixtures():
    rng = np.random.default_rng(61123); errors = []
    for n in (1, 2, 3, 6):
        q = np.exp(rng.normal(size=(2, n))); u = rng.normal(size=(2, n)); roles = rng.normal(size=(n, 4))
        a, p, _, _, c = records(q, u, roles)
        v = c + np.array([1., 0.])
        det = lambda x, y: x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]
        pair = sum((det(c[:, i], c[:, j]) for i in range(n) for j in range(i+1, n)), np.zeros(2)) / n**2
        triple = sum((det(v[:, j]-v[:, i], v[:, k]-v[:, i])
                      for i in range(n) for j in range(i+1, n) for k in range(j+1, n)), np.zeros(2)) / n**3
        b = sum((det(v[:, i], v[:, j]) for i in range(n) for j in range(i+1, n)), np.zeros(2)) / n**2
        rev = records(q[:, ::-1], u[:, ::-1], roles[::-1])[0]
        errors += [float(np.max(abs(a-pair))), float(np.max(abs(a-triple))),
                   float(np.max(abs(b-a-p))), float(np.max(abs(a+rev)))]
    q = np.array([[.1, .4, 1., .7]]); u = np.array([[2., -1., .5, 3.]])
    bins = {(i, j): [] for i in range(4) for j in range(4)}; all_area = []
    for order in itertools.permutations(range(4)):
        a = float(records(q[:, order], u[:, order], np.zeros((4, 4)))[0][0]); all_area.append(a)
        for position, token in enumerate(order):
            bins[token, position].append(a)
    permutation_error = max(abs(np.mean(v)) for v in bins.values())
    assert np.std(all_area) > 1e-4 and permutation_error < 1e-12
    for q0, u0 in [(np.ones((1, 5)), np.arange(5.)[None]),
                   (np.array([[1., 3., 1., 3., 1.]]), np.array([[2., -1., 2., -1., 2.]]))]:
        errors.append(float(abs(records(q0, u0, np.zeros((5, 4)))[0][0])))
    z = rng.normal(size=(3, 40)); f = centered(rng.normal(size=(3, 40))); y = np.array([0, 5, 39])
    prob = np.exp(z-z.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    g = prob.copy(); g[np.arange(3), y] -= 1
    def ce(v):
        m = v.max(1); return m + np.log(np.exp(v-m[:, None]).sum(1)) - v[np.arange(3), y]
    finite = (ce(z+1e-5*f)-ce(z-1e-5*f))/(2e-5)
    derivative_error = float(np.max(abs(finite-(g*f).sum(1))))
    assert max(errors) < 1e-11 and derivative_error < 1e-8
    return dict(status='PASS', pair_triple_trend_reversal_degeneracy_error=max(errors),
                conditional_permutation_error=permutation_error, ce_derivative_error=derivative_error,
                no_trainable_plugin_or_gradient_update=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'inputs', 'checkpoint', 'words', 'design', 'output', 'progress'):
        p.add_argument('--'+n, required=True, type=Path)
    p.add_argument('--attempt', required=True); args = p.parse_args()
    cfg = json.loads(args.design.read_bytes()); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'started.json').exists(), 'No duplicate diagnostic'
    assert digest(__file__) == cfg['code_sha256'] and digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.words) == cfg['words_sha256']
    assert digest(args.inputs/'discovery/manifest.json') == cfg['native_manifest_sha256']
    for n, h in cfg['source_hashes'].items():
        assert digest(args.source/n) == h
    words = json.loads(args.words.read_bytes()); assert len(words) == cfg['train_items'] == 8192
    ids = [w['id_hash'] for w in words]; assert len(set(ids)) == 8192
    groups = [hashlib.sha256(json.dumps(w['tokens'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() for w in words]
    calibration = np.array([int(hashlib.sha256(('ordered-mass-value-v1:'+g).encode()).hexdigest(), 16)%2 == 0 for g in groups])
    assessment = ~calibration
    assert calibration.sum() >= 1000 and assessment.sum() >= 1000
    split_sha = hashlib.sha256(json.dumps(dict(ids=ids, groups=groups, calibration=calibration.tolist()), sort_keys=True).encode()).hexdigest()
    # Split and input identities are fixed and recorded BEFORE the target file is opened.
    write(args.output/'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design),
        split_sha256=split_sha, calibration=int(calibration.sum()), assessment=int(assessment.sum())))
    signal.alarm(240); checks = fixtures(); started = time.time()
    import torch
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    native = RMEAHead('quantum').double().cuda().eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True)); native.requires_grad_(False)
    cache = args.inputs/'discovery/cache'
    data = {k: np.load(cache/('train-'+k+'.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets', 'targets')}
    assert len(data['targets']) == 58465 and len(data['offsets']) == 58466
    arrays = {k: [] for k in ('logits', 'area', 'trend', 'entity', 'broken_area', 'broken_trend', 'broken_entity', 'lengths')}
    parity = 0.; collection_started = time.time()
    with torch.inference_mode():
        for lo in range(0, 8192, 16):
            lengths = [int(data['offsets'][i+1]-data['offsets'][i]) for i in range(lo, lo+16)]
            h = torch.zeros(16, max(lengths), 1024, dtype=torch.float64, device='cuda')
            roles = torch.zeros(16, max(lengths), 4, dtype=torch.float64, device='cuda')
            mask = torch.zeros(16, max(lengths), dtype=torch.bool, device='cuda')
            for j, n in enumerate(lengths):
                aa, bb = map(int, data['offsets'][lo+j:lo+j+2])
                assert len(words[lo+j]['tokens']) == n
                h[j, :n] = torch.from_numpy(np.array(data['h'][aa:bb], dtype=np.float64, copy=True)).cuda()
                roles[j, :n] = torch.from_numpy(np.array(data['roles'][aa:bb], dtype=np.float64, copy=True)).cuda(); mask[j, :n] = True
            detail = native(h, roles, mask, details=True); bias = native.bias.cpu().numpy()
            z = detail['logits'].cpu().numpy(); probabilities = detail['probability'].cpu().numpy()
            utilities = detail['utilities'].cpu().numpy(); role_values = roles.cpu().numpy()
            for j, n in enumerate(lengths):
                q = probabilities[j, :, :n] + EPS; u = utilities[j, :, :n]; role = role_values[j, :n]
                a, p0, e, s, _ = records(q, u, role)
                parity = max(parity, float(np.max(abs(s+bias-z[j]))))
                order = sorted(range(n), key=lambda t: hashlib.sha256(('ordered-value-v1:'+ids[lo+j]+':'+str(t)).encode()).digest())
                if n > 1 and order == list(range(n)):
                    order = order[1:] + order[:1]
                broken_a, broken_p, broken_e, _, _ = records(q, u[:, order], role)
                for k, v in [('logits', z[j]), ('area', a), ('trend', p0), ('entity', e),
                             ('broken_area', broken_a), ('broken_trend', broken_p), ('broken_entity', broken_e), ('lengths', n)]:
                    arrays[k].append(v)
            write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                phase='ordered_native_records', current=lo+16, total=8192, unit='example',
                heartbeat_at=time.time(), observed_at=time.time()))
    assert parity < 1e-8
    arrays = {k: np.asarray(v) for k, v in arrays.items()}; collection_seconds = time.time()-collection_started
    labels = np.array(data['targets'][:8192], dtype=np.int64, copy=True)
    assert (labels >= 0).all() and (labels < 40).all()
    z = arrays['logits']; prob = np.exp(z-z.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    g = prob.copy(); g[np.arange(8192), labels] -= 1
    residual, fit, ranks = residualize(arrays['area'], z, arrays['trend'], arrays['entity'], arrays['lengths'], calibration)
    control, control_fit, control_ranks = residualize(arrays['broken_area'], z, arrays['broken_trend'], arrays['broken_entity'], arrays['lengths'], calibration)
    f, rms = unit_rms(residual); cf, crms = unit_rms(control)
    eligible = (rms > 1e-12) & (crms > 1e-12); f[~eligible] = 0; cf[~eligible] = 0
    assert np.max(abs(np.sqrt((f**2).mean(1))-np.sqrt((cf**2).mean(1)))) < 1e-10
    d, cd = (g*f).sum(1), (g*cf).sum(1)
    orientation = -float(np.sign(d[calibration].mean())); control_orientation = -float(np.sign(cd[calibration].mean()))
    d *= orientation; cd *= control_orientation
    raw, _ = unit_rms(arrays['area']); raw_d = (g*raw).sum(1)*orientation
    position, _ = unit_rms(arrays['trend']); position_d = (g*position).sum(1)
    position_orientation = -float(np.sign(position_d[calibration].mean())); position_d *= position_orientation
    b = bootstrap(np.column_stack([d[assessment], d[assessment]-cd[assessment]]),
                  np.asarray(groups)[assessment], cfg['bootstrap_replicates'], cfg['bootstrap_seed'])
    intervals = np.quantile(b, [.025, .975], axis=0)
    qualified = bool(eligible[calibration].mean() >= .99 and eligible[assessment].mean() >= .99)
    gates = dict(nonzero_orientation=orientation != 0 and control_orientation != 0, matched_feature_nondegenerate=qualified,
        raw_direction_consistent=float(raw_d[assessment].mean()) < 0,
        adjusted_negative_upper=float(intervals[1, 0]) < 0,
        paired_beats_broken_upper=float(intervals[1, 1]) < 0)
    arrays.update(labels=labels, calibration=calibration, groups=np.array(groups), residual=residual,
                  broken_residual=control, nuisance_fit=fit, broken_nuisance_fit=control_fit,
                  normalized_intact=f, normalized_broken=cf, derivative=d, broken_derivative=cd,
                  raw_derivative=raw_d, bootstrap=b, eligible=eligible)
    np.savez(args.output/'diagnostic.private.npz', **arrays)
    write(args.output/'fixtures.json', checks)
    summary = dict(status='completed', train_items=8192, relations=40, targets_read='TRAIN only after splitseal',
        valid_read=False, test_read=False, optimizer_updates=0, cos_writes=0,
        seconds=time.time()-started, collection_seconds=collection_seconds, native_replay_max_error=parity,
        calibration=int(calibration.sum()), assessment=int(assessment.sum()),
        unique_sentence_groups=len(set(groups)), grouping='Exact token-list hashes, NOT document/source IDs',
        split_sha256=split_sha, orientations=dict(intact=orientation, broken=control_orientation, position=position_orientation),
        eligible_counts=dict(calibration=int(eligible[calibration].sum()), assessment=int(eligible[assessment].sum())),
        assessment_derivatives=dict(intact=float(d[assessment].mean()), broken=float(cd[assessment].mean()),
            paired=float((d-cd)[assessment].mean()), raw_intact=float(raw_d[assessment].mean()),
            position_descriptive=float(position_d[assessment].mean())),
        bootstrap_percentile_intervals=dict(intact=list(map(float, intervals[:, 0])), paired=list(map(float, intervals[:, 1]))),
        assessment_class_contributions=list(map(float, (g*f*orientation)[assessment].mean(0))),
        nuisance_rank_range=[min(ranks+control_ranks), max(ranks+control_ranks)],
        gates=gates, gate='PASS_FOR_PLUGIN_PREPARATION_ONLY' if all(gates.values()) else 'FAIL_CLOSE_EXACT_PROFILE',
        warning='Exploratory TRAIN diagnostic on exposed checkpoint; approximate bootstrap only, no independent utility, newmath, quantumnecessity or L5',
        code_sha256=cfg['code_sha256'], design_sha256=digest(args.design),
        checkpoint_sha256=cfg['checkpoint_sha256'], source_hashes=cfg['source_hashes'],
        attempt=args.attempt, private_diagnostic_sha256=digest(args.output/'diagnostic.private.npz'), L5='NOT_ACCEPTED')
    write(args.output/'summary.json', summary)
    write(args.output/'RUN_COMPLETE', {n:digest(args.output/n) for n in ('summary.json','fixtures.json')})
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed',
        current=8192, total=8192, unit='example', heartbeat_at=time.time(), observed_at=time.time()))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
