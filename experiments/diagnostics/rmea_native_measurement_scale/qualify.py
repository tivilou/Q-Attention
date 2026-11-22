"""TRAIN-only scale planning for native decision-preserving overlap sampling."""
import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def sampling_moments(a, probability, leverage):
    """Exact law of X=a_t*Bernoulli(p_t)/(T*q_t), not a simulated sample."""
    a = np.asarray(a, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    t = a.shape[-1]
    assert a.shape == probability.shape and np.isfinite(a).all()
    assert (probability >= -1e-12).all() and (probability <= 1 + 1e-12).all()
    probability = np.clip(probability, 0, 1)
    if leverage:
        total = np.abs(a).sum(-1, keepdims=True)
        q = np.divide(np.abs(a), total, out=np.zeros_like(a), where=total > 0)
    else:
        q = np.full_like(a, 1 / t)
    values = np.divide(a, t * q, out=np.zeros_like(a), where=q > 0)
    mu = (a * probability).mean(-1)
    second = (q * probability * values**2).sum(-1)
    variance = second - mu**2
    assert (variance >= -1e-10 * np.maximum(1, second)).all()
    variance = np.maximum(variance, 0)
    lower = np.minimum(values.min(-1), 0)
    upper = np.maximum(values.max(-1), 0)
    width = upper - lower
    return dict(q=q, mean=mu, variance=variance, range=width,
                max_abs=np.max(np.abs(values), axis=-1))


def planning_width(variance, support_range, n, delta):
    """Population-variance substitution: NOT a confidence interval or guarantee."""
    log = np.log(2 / delta)
    return np.sqrt(2 * variance * log / n) + 7 * support_range * log / (3 * (n - 1))


def fixtures():
    a = np.array([0., -2., 1., 3.]); p = np.array([0., .25, .8, 1.])
    errors = []
    for leverage in (False, True):
        m = sampling_moments(a, p, leverage)
        outcomes, masses = [], []
        for i, q in enumerate(m['q']):
            if q > 0:
                outcomes.extend([0., a[i] / (4 * q)])
                masses.extend([q * (1 - p[i]), q * p[i]])
        x, w = np.array(outcomes), np.array(masses)
        mu = w @ x; variance = w @ (x - mu)**2
        errors += [abs(mu - m['mean']), abs(variance - m['variance'])]
        assert abs(m['q'].sum() - 1) < 1e-14
        if leverage:
            errors.append(abs(m['max_abs'] - np.abs(a).mean()))
    z = sampling_moments(np.zeros(4), p, True)
    assert z['variance'] == z['range'] == z['mean'] == 0
    assert max(errors) < 1e-12
    assert planning_width(np.array(1.), np.array(2.), 1024, .01 / 240) > 0
    return dict(status='PASS', enumerated_bernoulli_max_error=float(max(errors)),
                zero_coefficient_handling=True, mean_range_identity=True)


def quantiles(a):
    a = np.asarray(a)
    assert np.isfinite(a).all()
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    map(float, np.quantile(a, [0, .1, .5, .9, 1]))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'inputs', 'checkpoint', 'design', 'output', 'progress'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--attempt', required=True); args = p.parse_args()
    cfg = json.loads(args.design.read_bytes()); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'started.json').exists(), 'Never repeat the fixed qualification'
    assert digest(Path(__file__)) == cfg['code_sha256']
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for name, sha in cfg['source_hashes'].items():
        assert digest(args.source / name) == sha
    write(args.output / 'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design)))
    signal.alarm(180)
    import torch
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    native = RMEAHead('quantum').double().cuda().eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    native.requires_grad_(False)
    # No targets file is even opened. Original stored TRAIN order is the entire cohort rule.
    cache = args.inputs / 'discovery/cache'
    data = {k: np.load(cache / ('train-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets')}
    assert len(data['offsets']) == 58466 and cfg['train_items'] == 128
    checks = fixtures(); started = time.time(); replay_error = 0.; sign_error = 0.
    arrays = {k: [] for k in ('contrast', 'mean_denominator', 'top2_margin', 'leverage_mean',
                              'variance_uniform', 'variance_value', 'range_uniform', 'range_value')}
    with torch.inference_mode():
        for lo in range(0, 128, 16):
            indices = list(range(lo, lo + 16))
            lengths = [int(data['offsets'][i + 1] - data['offsets'][i]) for i in indices]
            h = torch.zeros(16, max(lengths), 1024, dtype=torch.float64, device='cuda')
            roles = torch.zeros(16, max(lengths), 4, dtype=torch.float64, device='cuda')
            mask = torch.zeros(16, max(lengths), dtype=torch.bool, device='cuda')
            for j, (i, n) in enumerate(zip(indices, lengths)):
                a, b = map(int, data['offsets'][i:i + 2])
                h[j, :n] = torch.from_numpy(np.array(data['h'][a:b], dtype=np.float64, copy=True)).cuda()
                roles[j, :n] = torch.from_numpy(np.array(data['roles'][a:b], dtype=np.float64, copy=True)).cuda()
                mask[j, :n] = True
            detail = native(h, roles, mask, details=True)
            probability = detail['probability'].cpu().numpy()
            utilities = detail['utilities'].cpu().numpy()
            logits = detail['logits'].cpu().numpy(); bias = native.bias.cpu().numpy()
            for j, n in enumerate(lengths):
                prob = probability[j, :, :n]; u = utilities[j, :, :n]
                reconstructed = ((prob + EPS) * u).sum(-1) / (prob + EPS).sum(-1) + bias
                replay_error = max(replay_error, float(np.max(np.abs(reconstructed - logits[j]))))
                ordered = np.sort(logits[j]); threshold = (ordered[-1] + ordered[-2]) / 2
                a = bias[:, None] + u - threshold
                denominator = (prob + EPS).mean(-1)
                contrast = (a * (prob + EPS)).mean(-1)
                sign_error = max(sign_error, float(np.max(np.abs(contrast / denominator - (logits[j] - threshold)))))
                arrays['contrast'].append(contrast)
                arrays['mean_denominator'].append(denominator)
                arrays['top2_margin'].append(ordered[-1] - ordered[-2])
                arrays['leverage_mean'].append(np.abs(a).mean(-1))
                for name, leverage in [('uniform', False), ('value', True)]:
                    m = sampling_moments(a, prob, leverage)
                    assert np.max(np.abs(m['mean'] + EPS * a.mean(-1) - contrast)) < 1e-10
                    arrays['variance_' + name].append(m['variance'])
                    arrays['range_' + name].append(m['range'])
            write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                phase='train_measurement_scale', current=lo + 16, total=128, unit='example',
                heartbeat_at=time.time(), observed_at=time.time()))
    assert replay_error < 1e-8 and sign_error < 1e-8
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez(args.output / 'scale.private.npz', **arrays)
    estimates = {}
    for name in ('uniform', 'value'):
        variance = arrays['variance_' + name]; width = arrays['range_' + name]
        rows = []
        for shots in cfg['shot_grid']:
            halfwidth = planning_width(variance, width, shots, cfg['delta'])
            resolvable = (np.abs(arrays['contrast']) > halfwidth).all(1) & (arrays['top2_margin'] > 0)
            rows.append(dict(shots_per_relation=shots, planning_all40_resolved=int(resolvable.sum()),
                planning_relations_resolved=int((np.abs(arrays['contrast']) > halfwidth).sum()),
                worst_relative_width=quantiles(np.max(np.divide(halfwidth, np.abs(arrays['contrast']),
                    out=np.full_like(halfwidth, 1e300), where=arrays['contrast'] != 0), axis=1))))
        estimates[name] = dict(variance=quantiles(variance), support_range=quantiles(width), grid=rows)
    ok = arrays['variance_uniform'] > 0
    ratio = arrays['variance_value'][ok] / arrays['variance_uniform'][ok]
    record = dict(status='completed', train_items=128, relations=40, labels_opened=False,
        valid_read=False, test_read=False, optimizer_updates=0, cos_writes=0,
        native_replay_max_error=replay_error, threshold_identity_max_error=sign_error, fixtures=checks,
        mean_denominator=quantiles(arrays['mean_denominator']), top2_margin=quantiles(arrays['top2_margin']),
        leverage_mean=quantiles(arrays['leverage_mean']), planning=estimates,
        value_over_uniform_variance=quantiles(ratio), variance_improved_relations=int((ratio < 1).sum()),
        positive_uniform_variance_relations=int(ok.sum()),
        optimistic_threshold='Midpoint of exact native top2 scores, planning only; not a deployable/controller threshold or optimal-threshold claim',
        warning='Population-variance planning widths only: NOT confidence intervals, sampled stopping times, guaranteed coverage or general algorithmic lower bounds',
        sampling_scope='Independent selected-token Bernoulli overlap; no physical hardware or simulator-shot experiment executed',
        seconds=time.time() - started, code_sha256=cfg['code_sha256'], design_sha256=digest(args.design),
        checkpoint_sha256=cfg['checkpoint_sha256'], source_hashes=cfg['source_hashes'], attempt=args.attempt,
        private_scale_sha256=digest(args.output / 'scale.private.npz'), L5='NOT_ACCEPTED')
    write(args.output / 'summary.json', record)
    write(args.output / 'RUN_COMPLETE', dict(summary_sha256=digest(args.output / 'summary.json')))
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed',
        current=128, total=128, unit='example', heartbeat_at=time.time(), observed_at=time.time()))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
