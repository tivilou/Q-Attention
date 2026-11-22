"""Exact TRAIN-only shared Pauli-measurement variance qualification."""
import argparse
import hashlib
import importlib.util
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


def write(p, v):
    pending = p.with_suffix('.pending')
    pending.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
    pending.replace(p)


def frame():
    """MSB tensor order; each of 27 bases is chosen with probability 1/27."""
    kets = [np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
            np.array([[1, 1j], [1, -1j]], complex) / np.sqrt(2),
            np.eye(2, dtype=complex)]
    vectors, snapshots = [], []
    for basis in itertools.product(range(3), repeat=3):
        for bits in itertools.product(range(2), repeat=3):
            v = np.array([1.], complex); snap = np.array([[1.]], complex)
            for axis, bit in zip(basis, bits):
                ket = kets[axis][bit]
                v = np.kron(v, ket)
                snap = np.kron(snap, 3 * np.outer(ket, ket.conj()) - np.eye(2))
            vectors.append(v); snapshots.append(snap)
    return np.array(vectors), np.array(snapshots)


def laws(psi, phi, vectors, snapshots):
    mass = np.abs(psi @ vectors.conj().T)**2 / 27
    z = np.einsum('ri,kij,rj->rk', phi.conj(), snapshots, phi)
    assert np.max(np.abs(z.imag)) < 1e-11
    readout = z.real
    assert np.max(np.abs(mass.sum(-1) - 1)) < 1e-10
    mean = mass @ readout.T
    second = mass @ (readout**2).T
    return mass, readout, mean, second


def fixtures(vectors, snapshots):
    rng = np.random.default_rng(7921)
    psi = rng.normal(size=(3, 8)) + 1j * rng.normal(size=(3, 8))
    psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
    phi = np.vstack([np.eye(8)[0], (np.eye(8)[0] + 1j * np.eye(8)[7]) / np.sqrt(2)])
    mass, readout, mean, second = laws(psi, phi, vectors, snapshots)
    reconstructed = np.einsum('tk,kij->tij', mass, snapshots)
    rho = np.einsum('ti,tj->tij', psi, psi.conj())
    inverse_error = float(np.max(np.abs(reconstructed - rho)))
    overlap_error = float(np.max(np.abs(mean - np.abs(psi @ phi.conj().T)**2)))
    # Independently enumerate a nonconstant weighted token/outcome law.
    a = np.array([[0., -2., 3.], [-1., .5, 2.]])
    values = a[:, :, None] * readout[:, None, :]
    weights = mass[None, :, :] / 3
    mu = (values * weights).sum((1, 2))
    enumerated_var = ((values - mu[:, None, None])**2 * weights).sum((1, 2))
    formula_var = (a**2 * second.T).mean(1) - mu**2
    variance_error = float(np.max(np.abs(enumerated_var - formula_var)))
    assert max(inverse_error, overlap_error, variance_error) < 1e-11
    return dict(status='PASS', inverse_channel_error=inverse_error,
                overlap_mean_error=overlap_error, weighted_variance_error=variance_error,
                bases=27, outcomes_per_basis=8, states=3, zero_coefficients=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'inputs', 'checkpoint', 'reference', 'design', 'output', 'progress'):
        p.add_argument('--' + n, required=True, type=Path)
    p.add_argument('--attempt', required=True); args = p.parse_args()
    cfg = json.loads(args.design.read_bytes()); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'started.json').exists(), 'No duplicate qualification'
    assert digest(__file__) == cfg['code_sha256']
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for n, h in cfg['source_hashes'].items():
        assert digest(args.source / n) == h
    assert digest(args.reference / 'qualify.py') == cfg['reference_code_sha256']
    assert digest(args.reference / 'output/scale.private.npz') == cfg['reference_scale_sha256']
    spec = importlib.util.spec_from_file_location('native_sampling_reference', args.reference / 'qualify.py')
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    previous = np.load(args.reference / 'output/scale.private.npz', allow_pickle=False)
    write(args.output / 'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design)))
    signal.alarm(180)
    import torch
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    native = RMEAHead('quantum').double().cuda().eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    native.requires_grad_(False)
    cache = args.inputs / 'discovery/cache'
    data = {k: np.load(cache / ('train-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets')}
    assert len(data['offsets']) == 58466 and cfg['train_items'] == 128
    vectors, snapshots = frame(); checks = fixtures(vectors, snapshots)
    arrays = {k: [] for k in ('contrast', 'variance_shared', 'variance_uniform', 'variance_value',
                              'range_shared', 'shared_proxy', 'uniform_proxy', 'value_proxy')}
    started = time.time(); parity = 0.; variance_floor_error = 0.
    with torch.inference_mode():
        for lo in range(0, 128, 16):
            indices = list(range(lo, lo + 16))
            lengths = [int(data['offsets'][i + 1] - data['offsets'][i]) for i in indices]
            h = torch.zeros(16, max(lengths), 1024, dtype=torch.float64, device='cuda')
            roles = torch.zeros(16, max(lengths), 4, dtype=torch.float64, device='cuda')
            mask = torch.zeros(16, max(lengths), dtype=torch.bool, device='cuda')
            for j, (i, n) in enumerate(zip(indices, lengths)):
                aa, bb = map(int, data['offsets'][i:i + 2])
                h[j, :n] = torch.from_numpy(np.array(data['h'][aa:bb], dtype=np.float64, copy=True)).cuda()
                roles[j, :n] = torch.from_numpy(np.array(data['roles'][aa:bb], dtype=np.float64, copy=True)).cuda()
                mask[j, :n] = True
            detail = native(h, roles, mask, details=True)
            logits = detail['logits'].cpu().numpy(); bias = native.bias.cpu().numpy()
            phi = detail['phi'].cpu().numpy()
            for j, n in enumerate(lengths):
                psi = detail['psi'][j, :n].cpu().numpy()
                prob = detail['probability'][j, :, :n].cpu().numpy()
                u = detail['utilities'][j, :, :n].cpu().numpy()
                ordered = np.sort(logits[j]); threshold = (ordered[-1] + ordered[-2]) / 2
                a = bias[:, None] + u - threshold
                contrast = (a * (prob + EPS)).mean(-1)
                mass, readout, mean, second = laws(psi, phi, vectors, snapshots)
                parity = max(parity, float(np.max(np.abs(mean.T - prob))))
                parity = max(parity, float(np.max(np.abs(contrast - previous['contrast'][lo + j]))))
                # Uniform token sampling is common to all relations; one state copy gives all readouts.
                variance = (a**2 * second.T).mean(-1) - (a * prob).mean(-1)**2
                variance_floor_error = min(variance_floor_error, float(variance.min()))
                assert (variance >= -1e-9).all()
                variance = np.maximum(variance, 0)
                amin, amax = a.min(-1), a.max(-1)
                hmin, hmax = readout.min(-1), readout.max(-1)
                corners = np.stack([amin*hmin, amin*hmax, amax*hmin, amax*hmax])
                support_range = corners.max(0) - corners.min(0)
                uniform = reference.sampling_moments(a, prob, False)['variance']
                value = reference.sampling_moments(a, prob, True)['variance']
                parity = max(parity, float(np.max(np.abs(uniform - previous['variance_uniform'][lo + j]))),
                             float(np.max(np.abs(value - previous['variance_value'][lo + j]))))
                # Never remove a zero-contrast case: it is an explicit ineligible row below.
                denom = contrast**2
                z = lambda v: np.divide(v, denom, out=np.full_like(v, np.inf), where=denom > 0)
                arrays['contrast'].append(contrast); arrays['variance_shared'].append(variance)
                arrays['variance_uniform'].append(uniform); arrays['variance_value'].append(value)
                arrays['range_shared'].append(support_range)
                arrays['shared_proxy'].append(float(z(variance).max()))
                arrays['uniform_proxy'].append(float(z(uniform).sum()))
                arrays['value_proxy'].append(float(z(value).sum()))
            write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                phase='train_shared_measurement_scale', current=lo + 16, total=128, unit='example',
                heartbeat_at=time.time(), observed_at=time.time()))
    assert parity < 1e-8
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    finite = np.isfinite(arrays['shared_proxy']) & np.isfinite(arrays['value_proxy']) & (arrays['value_proxy'] > 0)
    all_eligible = bool(finite.all()); comparisons = {}
    for name in ('uniform', 'value'):
        denominator = arrays[name + '_proxy']
        eligible = np.isfinite(arrays['shared_proxy']) & np.isfinite(denominator) & (denominator > 0)
        ratios = np.divide(arrays['shared_proxy'], denominator, out=np.full(128, np.inf), where=eligible)
        arrays['ratio_' + name] = ratios
        comparisons[name] = dict(eligible=int(eligible.sum()), improved=int((ratios < 1).sum()),
            ratio_quantiles=reference.quantiles(ratios) if bool(eligible.all()) else None)
    gate = all_eligible and comparisons['value']['ratio_quantiles']['median'] <= cfg['gate_median_ratio'] and comparisons['value']['improved'] >= cfg['gate_improved_cases']
    np.savez(args.output / 'scale.private.npz', **arrays)
    write(args.output / 'fixtures.json', checks)
    record = dict(status='completed', train_items=128, relations=40, labels_opened=False,
        valid_read=False, test_read=False, optimizer_updates=0, cos_writes=0, measurement_samples=0,
        seconds=time.time()-started, replay_and_mean_max_error=parity, variance_floor_error=variance_floor_error,
        comparisons=comparisons, shared_support_range=reference.quantiles(arrays['range_shared']),
        gate='PASS_FOR_FURTHER_DESIGN_ONLY' if gate else 'FAIL_CLOSE_EXACT_PROFILE',
        warning='Exact population variance and oracle threshold/allocation only; not finite-sample cost, confidence, measured advantage, predictive gain, novelty or L5',
        code_sha256=cfg['code_sha256'], design_sha256=digest(args.design),
        checkpoint_sha256=cfg['checkpoint_sha256'], source_hashes=cfg['source_hashes'], attempt=args.attempt,
        private_scale_sha256=digest(args.output / 'scale.private.npz'), L5='NOT_ACCEPTED')
    write(args.output / 'summary.json', record)
    write(args.output / 'RUN_COMPLETE', {n:digest(args.output/n) for n in ('summary.json','fixtures.json')})
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed',
        current=128, total=128, unit='example', heartbeat_at=time.time(), observed_at=time.time()))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
