"""Frozen native triangle-phase feasibility; stable model and evaluator unchanged."""
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
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def triangles(psi, phi, roles, mask):
    """NumPy complex128 mixed-span contraction, with original ordered role slots."""
    matrices = []
    present = np.ones(len(psi), dtype=bool)
    for slot in (2, 3):
        active = mask & (roles[..., slot] != 0)
        count = active.sum(1)
        present &= count > 0
        weights = active / np.maximum(count, 1)[:, None]
        matrices.append(np.einsum('bt,btd,bte->bde', weights, psi, psi.conj(), optimize=True))
    sphi, ophi = [np.einsum('bde,re->brd', matrix, phi, optimize=True) for matrix in matrices]
    a = np.einsum('rd,brd->br', phi.conj(), sphi).real
    b = np.einsum('rd,brd->br', phi.conj(), ophi).real
    c = np.einsum('brd,brd->br', sphi.conj(), ophi).imag
    features = np.stack((a, b, np.abs(c), c), -1)
    features[~present] = 0
    assert np.isfinite(features).all()
    return features, present


def fixtures(source):
    import torch
    sys.path.insert(0, str(source))
    from baselines.retacred.rmea import RMEAHead
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2, 5, 8)) + 1j * rng.normal(size=(2, 5, 8))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    phi = rng.normal(size=(40, 8)) + 1j * rng.normal(size=(40, 8))
    phi /= np.linalg.norm(phi, axis=-1, keepdims=True)
    roles = np.zeros((2, 5, 4)); roles[:, [0, 1], 2] = 1; roles[:, [2, 3], 3] = 1
    mask = np.ones((2, 5), dtype=bool); mask[:, 4] = False
    roles[:, 4, 2:] = 1  # A padded marked token must contribute nothing.
    f, _ = triangles(z, phi, roles, mask)
    explicit = np.zeros((2, 40), dtype=complex)
    for i in range(2):
        for s in (0, 1):
            for o in (2, 3):
                explicit[i] += ((phi.conj() @ z[i, s]) * np.vdot(z[i, s], z[i, o]) *
                                (z[i, o].conj() @ phi.T)) / 4
    pair_error = float(np.max(abs(f[..., 3] - explicit.imag)))
    swapped = roles.copy(); swapped[..., [2, 3]] = swapped[..., [3, 2]]
    reverse, _ = triangles(z, phi, swapped, mask)
    swap_error = float(np.max(abs(reverse - f[..., [1, 0, 2, 3]] * [1, 1, 1, -1])))
    empty = roles.copy(); empty[..., 2] = 0
    assert not triangles(z, phi, empty, mask)[0].any()
    torch.manual_seed(0); torch.set_num_threads(1)
    native = RMEAHead('quantum').double().eval()
    h = torch.randn(2, 5, 1024, dtype=torch.float64)
    role_t = torch.from_numpy(roles); mask_t = torch.from_numpy(mask)
    with torch.inference_mode():
        before = native(h, role_t, mask_t, details=True)
        native.prepare.weight[1::2].neg_(); native.prepare.bias[1::2].neg_()
        native.effect[..., 1].neg_()
        after = native(h, role_t, mask_t, details=True)
    logit_error = float((before['logits'] - after['logits']).abs().max())
    prob_error = float((before['probability'] - after['probability']).abs().max())
    fb, _ = triangles(before['psi'].numpy(), before['phi'].numpy(), roles, mask)
    fa, _ = triangles(after['psi'].numpy(), after['phi'].numpy(), roles, mask)
    conjugation_error = float(np.max(abs(fa - fb * [1, 1, 1, -1])))
    assert max(pair_error, swap_error, logit_error, prob_error, conjugation_error) < 1e-11
    return dict(pair_error=pair_error, slot_swap_error=swap_error, native_logit_error=logit_error,
                native_probability_error=prob_error, conjugation_error=conjugation_error,
                empty_span_zero=True, status='PASS')


def collect(args, cfg):
    import torch
    started = time.time(); signal.alarm(1800)
    assert not (args.output / 'features.private.npz').exists(), 'Never repeat the frozen collection'
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for name, sha in cfg['source_hashes'].items():
        assert digest(args.source / name) == sha
    assert digest(Path(__file__)) == cfg['code_sha256']
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    fixture = fixtures(args.source)
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead
    native = RMEAHead('quantum').double().cuda().eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0', weights_only=True))
    native.requires_grad_(False)
    cache = args.inputs / 'discovery/cache'; arrays = {}; empty = {}
    processed = 0; total = cfg['train_items'] + cfg['valid_items']
    with torch.inference_mode():
        for split, count in [('train', cfg['train_items']), ('valid', cfg['valid_items'])]:
            data = {k: np.load(cache / (split + '-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
                    for k in ('h', 'roles', 'offsets', 'targets')}
            assert len(data['targets']) == dict(train=58465, valid=19584)[split]
            logits = []; features = []; missing = 0
            for lo in range(0, count, 16):
                indices = list(range(lo, min(lo + 16, count)))
                lengths = [int(data['offsets'][i + 1] - data['offsets'][i]) for i in indices]
                h = torch.zeros(len(indices), max(lengths), 1024, dtype=torch.float64, device='cuda')
                roles = torch.zeros(len(indices), max(lengths), 4, dtype=torch.float64, device='cuda')
                mask = torch.zeros(len(indices), max(lengths), dtype=torch.bool, device='cuda')
                for j, (i, n) in enumerate(zip(indices, lengths)):
                    a, b = map(int, data['offsets'][i:i + 2])
                    h[j, :n] = torch.from_numpy(np.array(data['h'][a:b], dtype=np.float64, copy=True)).cuda()
                    roles[j, :n] = torch.from_numpy(np.array(data['roles'][a:b], dtype=np.float64, copy=True)).cuda()
                    mask[j, :n] = True
                detail = native(h, roles, mask, details=True)
                f, present = triangles(detail['psi'].cpu().numpy(), detail['phi'].cpu().numpy(),
                                       roles.cpu().numpy(), mask.cpu().numpy())
                logit = detail['logits'].cpu().numpy()
                assert np.isfinite(logit).all()
                logits.append(logit); features.append(f); missing += int((~present).sum())
                processed += len(indices)
                write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                      phase='triangle_features_' + split, current=processed, total=total, unit='example',
                      observed_at=time.time(), heartbeat_at=time.time()))
            arrays[split + '_logits'] = np.concatenate(logits)
            arrays[split + '_features'] = np.concatenate(features)
            arrays[split + '_targets'] = np.array(data['targets'][:count], copy=True)
            empty[split] = missing
    np.savez(args.output / 'features.private.npz', **arrays)
    record = dict(status='collected', count=dict(train=cfg['train_items'], valid=cfg['valid_items']),
                  empty_spans=empty, seconds=time.time() - started, fixtures=fixture,
                  checkpoint_sha256=cfg['checkpoint_sha256'], design_sha256=digest(args.design),
                  source_hashes=cfg['source_hashes'], code_sha256=cfg['code_sha256'],
                  feature_sha256=digest(args.output / 'features.private.npz'), test_read=False,
                  backbone_updates=0, cos_writes=0, dtype='float64/complex128', attempt=args.attempt)
    write(args.output / 'collection.json', record); print(json.dumps(record), flush=True)


def objective(theta, features, base, targets, penalty):
    from scipy.special import logsumexp
    coeff = theta.reshape(40, 4)
    logits = base + np.einsum('nrf,rf->nr', features, coeff, optimize=True)
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logs); p[np.arange(len(targets)), targets] -= 1
    loss = -float(logs[np.arange(len(targets)), targets].mean()) + penalty * float(theta @ theta)
    grad = np.einsum('nr,nrf->rf', p / len(targets), features, optimize=True) + 2 * penalty * coeff
    return loss, grad.ravel()


def fit(args, cfg):
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    sys.path.insert(0, str(args.source))
    from baselines.retacred.common import metrics
    signal.alarm(900); started = time.time()
    assert not (args.output / 'summary.json').exists(), 'Never repeat a frozen fit'
    collection = json.loads((args.output / 'collection.json').read_bytes())
    assert collection['design_sha256'] == digest(args.design)
    assert digest(Path(__file__)) == cfg['code_sha256']
    for name, sha in cfg['source_hashes'].items():
        assert digest(args.source / name) == sha
    assert digest(args.output / 'features.private.npz') == collection['feature_sha256']
    with np.load(args.output / 'features.private.npz', allow_pickle=False) as data:
        train_base = data['train_logits']; valid_base = data['valid_logits']
        train = data['train_features']; valid = data['valid_features']
        y = data['train_targets']; vy = data['valid_targets']
    assert len(y) == cfg['train_items'] and len(vy) == cfg['valid_items']
    labels = ['no_relation'] + ['class_' + str(i) for i in range(1, 40)]

    def evaluate(logits):
        ce = float((logsumexp(logits, axis=1) - logits[np.arange(len(vy)), vy]).mean())
        result = metrics(vy.tolist(), logits.argmax(1).tolist(), labels, ce)
        result['positive39_macro_f1'] = float(np.mean([r['f1'] for r in result['per_class'][1:]]))
        return result

    # A central finite difference validates the tiny-head objective, not training/resume infrastructure.
    rng = np.random.default_rng(0); tiny = rng.normal(size=(7, 40, 4)); tb = rng.normal(size=(7, 40))
    ty = np.arange(7); t = rng.normal(size=160) * .01; direction = rng.normal(size=160)
    grad = objective(t, tiny, tb, ty, cfg['penalty'])[1]
    fd = (objective(t + 1e-6 * direction, tiny, tb, ty, cfg['penalty'])[0] -
          objective(t - 1e-6 * direction, tiny, tb, ty, cfg['penalty'])[0]) / 2e-6
    grad_error = abs(fd - float(grad @ direction)); assert grad_error < 1e-7
    arms = {}; weights = {}
    for name in ('cyclic_phase', 'role_sign_control'):
        tf = train.copy(); vf = valid.copy()
        if name == 'role_sign_control':
            for values in (tf, vf):
                values[..., 3] = np.sign(values[..., 0] - values[..., 1]) * values[..., 2]
        scale = np.sqrt(np.mean(tf * tf, axis=0)); scale[scale == 0] = 1
        tf /= scale; vf /= scale
        result = minimize(objective, np.zeros(160), args=(tf, train_base, y, cfg['penalty']),
                          method='L-BFGS-B', jac=True,
                          options=dict(maxiter=cfg['max_iterations'], gtol=1e-8, ftol=1e-15, maxls=40))
        loss, grad = objective(result.x, tf, train_base, y, cfg['penalty'])
        norm = float(np.max(abs(grad)))
        predictions = valid_base + np.einsum('nrf,rf->nr', vf, result.x.reshape(40, 4), optimize=True)
        assert np.isfinite(predictions).all() and np.isfinite(result.x).all()
        arms[name] = dict(metrics=evaluate(predictions), optimizer_status=int(result.status),
                          optimizer_message=str(result.message), iterations=int(result.nit), objective=loss,
                          gradient_inf=norm, convergence_gate=bool(norm <= cfg['gradient_inf']),
                          parameters=160, train_scale_zero_columns=int(np.count_nonzero(np.mean(tf*tf, axis=0) == 0)))
        weights[name + '_theta'] = result.x; weights[name + '_scale'] = scale
    native = evaluate(valid_base); candidate = arms['cyclic_phase']['metrics']; contrasts = {}
    for name, reference in [('native', native), ('role_sign_control', arms['role_sign_control']['metrics'])]:
        delta_ce = reference['loss'] - candidate['loss']
        delta_macro = candidate['positive39_macro_f1'] - reference['positive39_macro_f1']
        contrasts[name] = dict(ce_improvement=delta_ce, positive39_macro_improvement=delta_macro,
                               ce_gate=delta_ce >= cfg['ce_gain'], macro_gate=delta_macro >= cfg['macro_gain'])
    converged = all(a['convergence_gate'] for a in arms.values())
    gate = converged and all(v['ce_gate'] and v['macro_gate'] for v in contrasts.values())
    record = dict(status='completed' if converged else 'numerically_unresolved', promotion_gate=gate,
                  native=native, arms=arms, contrasts=contrasts, collection=collection,
                  objective_gradient_fixture_error=grad_error, fit_seconds=time.time()-started,
                  L5='NOT_ACCEPTED', test_read=False, backbone_updates=0, cos_writes=0,
                  claim_scope='Single fixed native checkpoint, first8192TRAIN/exposed allVALID incremental tiny-readout feasibility; no original-L4 inheritance, independent confirmation, new Bargmann mathematics or L5.')
    np.savez(args.output / 'readout.private.npz', **weights)
    write(args.output / 'summary.json', record)
    write(args.output / 'RUN_COMPLETE', dict(summary_sha256=digest(args.output / 'summary.json')))
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase=record['status'],
          current=2, total=2, unit='readout_fit', observed_at=time.time(), heartbeat_at=time.time()))
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('collect', 'fit'))
    for name in ('source', 'inputs', 'checkpoint', 'design', 'output', 'progress'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--attempt', required=True); args = p.parse_args()
    cfg = json.loads(args.design.read_bytes()); args.output.mkdir(parents=True, exist_ok=True)
    (collect if args.stage == 'collect' else fit)(args, cfg)


if __name__ == '__main__':
    main()
