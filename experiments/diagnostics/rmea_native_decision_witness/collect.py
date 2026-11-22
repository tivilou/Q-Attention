"""One frozen native VALID inference pass; no TEST, fitting, or model writes."""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from witness import decision_operator


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    t = path.with_suffix('.pending')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'inputs', 'checkpoint', 'design', 'output', 'progress'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attempt', required=True)
    args = p.parse_args(); cfg = json.loads(args.design.read_bytes())
    signal.alarm(1800); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'operators.npz').exists(), 'Never reselect an existing analysis'
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for name, pin in cfg['source_hashes'].items():
        assert digest(args.source / name) == pin
    for name, pin in cfg['code_hashes'].items():
        assert digest(Path(__file__).parent / name) == pin
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    assert EPS == 1e-6
    cache = args.inputs / 'discovery/cache'
    train = np.load(cache / 'train-targets.npy', allow_pickle=False)
    valid = np.load(cache / 'valid-targets.npy', allow_pickle=False)
    assert train.shape == (58465,) and valid.shape == (19584,)
    counts = np.bincount(train, minlength=40)
    target = int(np.argmax(counts[1:]) + 1)  # Existing index0 is no_relation.
    data = {k: np.load(cache / ('valid-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets')}
    metadata = json.loads((cache / 'metadata.json').read_bytes())['valid']
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    native = RMEAHead('quantum').double().to(device).eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    native.requires_grad_(False)
    groups = [[], []]; rows = []; matrices = []; errors = []; processed = 0; started = time.time()

    def progress(stage):
        save(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase=stage,
             current=processed, total=19584, unit='example', observed_at=time.time(), heartbeat_at=time.time(),
             selected_groups=[len(x) for x in groups]))

    progress('native_valid_inference')
    with torch.inference_mode():
        for lo in range(0, len(valid), 16):
            indices = list(range(lo, min(lo + 16, len(valid))))
            lengths = [int(data['offsets'][i + 1] - data['offsets'][i]) for i in indices]
            h = torch.zeros(len(indices), max(lengths), 1024, dtype=torch.float64, device=device)
            roles = torch.zeros(len(indices), max(lengths), 4, dtype=torch.float64, device=device)
            mask = torch.zeros(len(indices), max(lengths), dtype=torch.bool, device=device)
            for j, (i, n) in enumerate(zip(indices, lengths)):
                a, b = map(int, data['offsets'][i:i + 2])
                h[j, :n] = torch.from_numpy(np.array(data['h'][a:b], dtype=np.float64, copy=True)).to(device)
                roles[j, :n] = torch.from_numpy(np.array(data['roles'][a:b], dtype=np.float64, copy=True)).to(device)
                mask[j, :n] = True
            detail = native(h, roles, mask, details=True)
            logits = detail['logits'].cpu().numpy()
            assert np.isfinite(logits).all()
            for j, i in enumerate(indices):
                order = np.argsort(-logits[j], kind='stable')
                winner, second, third = map(int, order[:3])
                if winner != int(valid[i]) or logits[j, winner] <= logits[j, second]:
                    continue
                group = 0 if winner == target else 1
                if group == 1 and (second != target or logits[j, second] <= logits[j, third]):
                    continue
                if len(groups[group]) == 128:
                    continue
                rival = max(logits[j, k] for k in range(40) if k != target)
                sign = 1 if group == 0 else -1
                psi = detail['psi'][j].cpu().numpy()
                u = detail['utilities'][j, target].cpu().numpy()
                m = mask[j].cpu().numpy()
                assert np.max(np.abs(np.sum(abs(psi[m])**2, axis=1) - 1)) < 1e-10
                phi = detail['phi'][target].cpu().numpy()
                operator, normalized, scale = decision_operator(psi, u, float(native.bias[target]),
                                                                float(rival), sign, m, EPS)
                denominator = float(detail['normalizer'][j, target, 0])
                direct = sign * (float(logits[j, target]) - float(rival))
                replay = float(np.vdot(phi, operator @ phi).real) / denominator
                error = abs(direct - replay)
                if error > 1e-9 + 1e-8 * abs(direct) or direct <= 0:
                    raise ValueError('Native decision replay failed; no dropping/reselection')
                row = dict(index=i, id_hash=metadata[i]['id_hash'], group=group, native_margin=direct,
                           replay_margin=replay, replay_error=error, operator_scale=scale)
                groups[group].append(len(rows)); rows.append(row); matrices.append(normalized); errors.append(error)
            processed = indices[-1] + 1; progress('native_valid_inference')
            if all(len(g) == 128 for g in groups):
                break
    supported = all(len(g) >= 32 for g in groups)
    summary = dict(status='collected' if supported else 'under_supported', selected_relation=target,
                   train_positive_counts=counts[1:].tolist(), group_sizes=[len(g) for g in groups],
                   processed_valid=processed, replay_max_error=max(errors, default=0), seconds=time.time() - started,
                   checkpoint_sha256=cfg['checkpoint_sha256'], design_sha256=digest(args.design),
                   source_hashes=cfg['source_hashes'], code_hashes=cfg['code_hashes'], dtype='float64/complex128',
                   optimizer_updates=0, test_read=False, L5='NOT_ACCEPTED')
    save(args.output / 'selection.private.json', dict(summary=summary, cases=rows))
    if supported:
        weights = np.array([.5 / len(groups[row['group']]) for row in rows])
        np.savez(args.output / 'operators.npz', operators=np.array(matrices), weights=weights)
        summary['operators_sha256'] = digest(args.output / 'operators.npz')
    save(args.output / 'collection.json', summary); progress('collected' if supported else 'under_supported')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
