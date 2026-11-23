"""Frozen native two-effect screen; compare mixing before/after normalization."""
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


def initial_effect(native):
    direction = np.random.default_rng(0).normal(size=(40, 12))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    delta = direction.reshape(40, 2, 3, 2) * .1
    raw = native.detach().cpu().numpy()
    return np.stack((raw + delta, raw - delta), axis=1)


def moments(detail, mask):
    import torch
    from baselines.retacred.rmea import EPS
    psi, u = detail['psi'], detail['utilities']
    rho = psi[..., :, None] * psi.conj()[..., None, :]
    eye = torch.eye(8, dtype=psi.dtype, device=psi.device)
    weighted = (rho + EPS * eye) * mask[..., None, None]
    c = weighted.sum(1)
    b = torch.einsum('brt,btde->brde', u.to(psi.dtype), weighted)
    return c, b


def components(raw, c, b):
    import torch
    import math
    from baselines.retacred.rmea import circuit
    phi = circuit(math.pi * torch.tanh(raw))
    z = torch.einsum('rlx,bxy,rly->brl', phi.conj(), c, phi).real
    n = torch.einsum('rlx,brxy,rly->brl', phi.conj(), b, phi).real
    assert torch.isfinite(n).all() and torch.isfinite(z).all() and (z > 0).all()
    return n, z


def readout(raw, c, b, bias, before):
    n, z = components(raw, c, b)
    return (n.sum(-1) / z.sum(-1) if before else (n / z).mean(-1)) + bias


def direct(raw, detail, mask, bias, before):
    import torch
    import math
    from baselines.retacred.rmea import circuit, EPS
    phi = circuit(math.pi * torch.tanh(raw))
    p = torch.einsum('rld,btd->brlt', phi.conj(), detail['psi']).abs().square()
    weights = (p + EPS) * mask[:, None, None, :]
    if before:
        weights = weights.mean(2)
        alpha = weights / weights.sum(-1, keepdim=True)
    else:
        alpha = (weights / weights.sum(-1, keepdim=True)).mean(2)
    return (alpha * detail['utilities']).sum(-1) + bias


def changed_mechanism_checks(native, h, roles, mask, targets):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        detail = native(h, roles, mask, details=True)
        c, b = moments(detail, mask)
    assert len(set(mask.sum(-1).tolist())) > 1 and not mask.all()
    seed = initial_effect(native.effect)
    errors, gradient_errors, updates = {}, {}, {}
    coincident = torch.stack((native.effect, native.effect), 1)
    for before in (True, False):
        name = 'before' if before else 'after'
        errors[name + '_coincident_native'] = float((readout(coincident, c, b, native.bias, before) - detail['logits']).abs().max())
        raw = torch.tensor(seed, dtype=torch.float64, device=h.device, requires_grad=True)
        actual = readout(raw, c, b, native.bias, before)
        reference = direct(raw, detail, mask, native.bias, before)
        errors[name + '_direct'] = float((actual - reference).abs().max())
        ga, = torch.autograd.grad(F.cross_entropy(actual, targets), raw, retain_graph=True)
        gb, = torch.autograd.grad(F.cross_entropy(reference, targets), raw)
        gradient_errors[name] = float((ga - gb).abs().max())
        optimizer = torch.optim.Adam([raw], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        loss = F.cross_entropy(readout(raw, c, b, native.bias, before), targets)
        optimizer.zero_grad(); loss.backward()
        assert torch.isfinite(raw.grad).all() and raw.grad.norm() > 0
        old = raw.detach().clone(); optimizer.step()
        updates[name] = float((raw - old).abs().max())
        assert torch.isfinite(raw).all() and updates[name] > 0
    raw = torch.tensor(seed, dtype=torch.float64, device=h.device)
    n, z = components(raw, c, b)
    lhs = readout(raw, c, b, native.bias, True) - readout(raw, c, b, native.bias, False)
    rhs = (z[..., 0] - z[..., 1]) * (n[..., 0] / z[..., 0] - n[..., 1] / z[..., 1]) / (2 * z.sum(-1))
    errors['normalization_identity'] = float((lhs - rhs).abs().max())
    assert max(errors.values()) < 1e-8 and max(gradient_errors.values()) < 1e-8
    return dict(status='PASS', errors=errors, gradient_errors=gradient_errors, real_update_deltas=updates,
                variable_lengths=mask.sum(-1).tolist(), fixture_updates_per_arm=1, fixture_state_discarded=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'inputs', 'checkpoint', 'design', 'output', 'progress'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attempt', required=True); args = p.parse_args()
    cfg = json.loads(args.design.read_bytes()); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'started.json').exists(), 'No automatic scientific restart or rescue'
    assert digest(Path(__file__)) == cfg['code_sha256']
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for name, sha in cfg['source_hashes'].items():
        assert digest(args.source / name) == sha
    write(args.output / 'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design)))
    signal.alarm(1800)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.common import metrics
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    native = RMEAHead('quantum').double().cuda().eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    native.requires_grad_(False)
    cache = args.inputs / 'discovery/cache'
    data = {split: {k: np.load(cache / (split + '-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
                   for k in ('h', 'roles', 'offsets', 'targets')} for split in ('train', 'valid')}
    assert len(data['train']['targets']) == 58465 and len(data['valid']['targets']) == 19584
    counts = dict(train=cfg['train_items'], valid=cfg['valid_items'])
    assert counts == dict(train=8192, valid=19584)
    started = time.time()

    def batch(split, indices):
        d = data[split]; lengths = [int(d['offsets'][i + 1] - d['offsets'][i]) for i in indices]
        h = torch.zeros(len(indices), max(lengths), 1024, dtype=torch.float64, device='cuda')
        roles = torch.zeros(len(indices), max(lengths), 4, dtype=torch.float64, device='cuda')
        mask = torch.zeros(len(indices), max(lengths), dtype=torch.bool, device='cuda')
        for j, (i, n) in enumerate(zip(indices, lengths)):
            a, b = map(int, d['offsets'][i:i + 2])
            h[j, :n] = torch.from_numpy(np.array(d['h'][a:b], dtype=np.float64, copy=True)).cuda()
            roles[j, :n] = torch.from_numpy(np.array(d['roles'][a:b], dtype=np.float64, copy=True)).cuda()
            mask[j, :n] = True
        return h, roles, mask

    h, roles, mask = batch('train', list(range(8)))
    fixture = changed_mechanism_checks(native, h, roles, mask,
        torch.tensor(np.array(data['train']['targets'][:8], dtype=np.int64), device='cuda'))
    write(args.output / 'fixtures.json', fixture)
    buffers, base, labels = {}, {}, {}
    processed = 0; parity = 0.
    with torch.no_grad():
        for split, count in counts.items():
            c_np = np.lib.format.open_memmap(args.output / (split + '-C.private.npy'), mode='w+', dtype=np.complex128, shape=(count, 8, 8))
            b_np = np.lib.format.open_memmap(args.output / (split + '-B.private.npy'), mode='w+', dtype=np.complex128, shape=(count, 40, 8, 8))
            original = []
            for lo in range(0, count, 16):
                hi = min(lo + 16, count); h, roles, mask = batch(split, list(range(lo, hi)))
                detail = native(h, roles, mask, details=True); c, b = moments(detail, mask)
                coincident = torch.stack((native.effect, native.effect), 1)
                parity = max(parity, float((readout(coincident, c, b, native.bias, True) - detail['logits']).abs().max()))
                c_np[lo:hi] = c.cpu().numpy(); b_np[lo:hi] = b.cpu().numpy()
                original.append(detail['logits'].cpu().numpy()); processed += hi - lo
                write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                    phase='moments_' + split, current=processed, total=sum(counts.values()), unit='example',
                    heartbeat_at=time.time(), observed_at=time.time()))
            assert parity < 1e-8
            c_np.flush(); b_np.flush()
            # This small fixed cache fits one GPU; no token-state backpropagation or backbone work.
            buffers[split] = (torch.from_numpy(np.array(c_np)).cuda(), torch.from_numpy(np.array(b_np)).cuda())
            del c_np, b_np
            base[split] = np.concatenate(original)
            labels[split] = np.array(data[split]['targets'][:count], dtype=np.int64, copy=True)
    collect_seconds = time.time() - started
    y = torch.tensor(labels['train'], dtype=torch.int64, device='cuda')
    seed = torch.tensor(initial_effect(native.effect), dtype=torch.float64, device='cuda')
    arms = {'before': torch.nn.Parameter(seed.clone()), 'after': torch.nn.Parameter(seed.clone())}
    assert all(raw.numel() == 960 for raw in arms.values())
    optimizer = torch.optim.Adam(list(arms.values()), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
    updates = 0; orders = np.random.default_rng(13); train_started = time.time()
    order_hashes = []; history = []
    for epoch in range(2):
        order = orders.permutation(counts['train']); order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        totals = {name: 0. for name in arms}
        for lo in range(0, counts['train'], 32):
            idx = torch.tensor(order[lo:lo + 32], device='cuda')
            c, b = (v[idx] for v in buffers['train']); optimizer.zero_grad()
            losses = {name: F.cross_entropy(readout(raw, c, b, native.bias, name == 'before'), y[idx])
                      for name, raw in arms.items()}
            loss = sum(losses.values()); assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(raw.grad).all() for raw in arms.values())
            optimizer.step(); updates += 1
            assert all(torch.isfinite(raw).all() for raw in arms.values())
            for name in arms:
                totals[name] += float(losses[name]) * len(idx)
            if updates % 16 == 0:
                write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                    phase='train_both_effect_heads', current=updates, total=512, unit='updates_per_arm',
                    heartbeat_at=time.time(), observed_at=time.time()))
        history.append(dict(epoch=epoch + 1, loss={name: v / counts['train'] for name, v in totals.items()}))
    assert updates == 512
    train_seconds = time.time() - train_started

    def evaluate(logits):
        target = torch.tensor(labels['valid'], device='cuda')
        ce = float(F.cross_entropy(torch.tensor(logits, device='cuda'), target))
        result = metrics(labels['valid'].tolist(), logits.argmax(1).tolist(),
                         ['no_relation'] + ['class_' + str(i) for i in range(1, 40)], ce)
        result['positive39_macro_f1'] = float(np.mean([r['f1'] for r in result['per_class'][1:]]))
        return result

    results = {}
    with torch.no_grad():
        for name, raw in arms.items():
            predictions = []
            for lo in range(0, counts['valid'], 128):
                c, b = (v[lo:lo + 128] for v in buffers['valid'])
                predictions.append(readout(raw, c, b, native.bias, name == 'before').cpu().numpy())
            logits = np.concatenate(predictions); assert np.isfinite(logits).all()
            results[name] = evaluate(logits)
    reference = evaluate(base['valid']); candidate = results['before']; contrasts = {}
    for name, control in [('native', reference), ('after', results['after'])]:
        ce = control['loss'] - candidate['loss']
        macro = candidate['positive39_macro_f1'] - control['positive39_macro_f1']
        contrasts[name] = dict(ce_improvement=ce, positive39_macro_improvement=macro,
            ce_gate=ce >= cfg['ce_gain'], macro_gate=macro >= cfg['macro_gain'])
    np.savez(args.output / 'final-effects.private.npz', **{k: v.detach().cpu().numpy() for k, v in arms.items()})
    result = dict(status='completed', promotion_gate=all(v['ce_gate'] and v['macro_gate'] for v in contrasts.values()),
        native=reference, arms=results, contrasts=contrasts, train_history=history, fixtures=fixture,
        native_moment_parity_max_error=parity, updates_per_arm=updates, parameters_per_arm=960,
        collection_seconds=collect_seconds, training_seconds=train_seconds, seconds=time.time() - started,
        train_order_sha256=order_hashes, counts=counts, attempt=args.attempt,
        checkpoint_sha256=cfg['checkpoint_sha256'], design_sha256=digest(args.design), code_sha256=cfg['code_sha256'],
        source_hashes=cfg['source_hashes'], final_effects_sha256=digest(args.output / 'final-effects.private.npz'),
        test_read=False, backbone_updates=0, cos_writes=0, L5='NOT_ACCEPTED',
        claim_scope='One exposed TRAIN/VALID fixed-native tiny-head feasibility, not independent confirmation, full-model utility, original L4 inheritance, mixed-state novelty, quantum exclusivity or L5.')
    write(args.output / 'summary.json', result)
    write(args.output / 'RUN_COMPLETE', dict(summary_sha256=digest(args.output / 'summary.json')))
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed', current=512,
        total=512, unit='updates_per_arm', heartbeat_at=time.time(), observed_at=time.time()))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
