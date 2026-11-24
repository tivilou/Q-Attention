"""Exploratory CPU-only coordinate diagnostic of immutable selected heads.

This computes pi*tanh's derivative, not a model gradient or a causal explanation.
No forward circuit, logits, optimizer, labels, validation, test, or disk writes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib
import json
import resource
import sys
import time
import numpy as np
import torch

packet = json.load(sys.stdin)
cfg, allocation = packet['design'], packet['allocation']
started = time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
os.nice(10)
torch.set_num_threads(1)
root = Path('/hy-tmp/Q-Attention')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

permit = Path(allocation['permit_path'])
assert digest(permit) == allocation['permit_sha256']
p = json.loads(permit.read_bytes())
for field in ('allocation_id', 'generation', 'project_id', 'host_id', 'run_id', 'runner_attempt_id'):
    assert p[field] == allocation[field]
for name, expected in cfg['input_hashes'].items():
    assert digest(root / name) == expected, name

prior = root / 'control/historical-type-replay-20261123'
provenance = json.loads((prior / 'model-provenance.json').read_bytes())
cache = root / 'inputs/native-baseline/discovery/cache'
meta = json.loads((cache / 'metadata.json').read_bytes())['train']
assert len(meta) == 58465 and len({r['id_hash'] for r in meta}) == 58465
indices = sorted(range(len(meta)), key=lambda i: meta[i]['id_hash'])[:cfg['sample_rows']]
data = {k: np.load(cache / ('train-' + k + '.npy'), mmap_mode='r', allow_pickle=False)
        for k in ('h', 'roles', 'offsets')}
offsets = data['offsets']
assert offsets.shape == (58466,) and offsets[0] == 0
assert offsets[-1] == len(data['h']) == len(data['roles'])
assert data['h'].shape[1:] == (1024,) and data['roles'].shape[1:] == (4,)
actual_inputs = hashlib.sha256()
blocks = []
for i in indices:
    left, right = map(int, offsets[i:i + 2])
    assert 0 <= left < right <= len(data['h']) and right - left == meta[i]['words']
    actual_inputs.update(meta[i]['id_hash'].encode())
    for kind in ('h', 'roles'):
        actual_inputs.update(np.ascontiguousarray(data[kind][left:right]).tobytes())
    blocks.append(np.concatenate((data['h'][left:right].astype(np.float64),
                                  data['roles'][left:right].astype(np.float64)), axis=1))
x = np.concatenate(blocks)
assert np.isfinite(x).all()

def stats(z):
    assert np.isfinite(z).all()
    # sech^2(z), expressed stably even for large |z|.
    e = np.exp(-2 * np.abs(z))
    d = 4 * e / (1 + e) ** 2
    return dict(coordinates=z.size, derivative_relative_to_pi_quantiles={str(q): float(np.quantile(d, q))
                for q in (0, .01, .1, .5, .9, .99, 1)},
                relative_derivative_below_001=float(np.mean(d < .01)),
                relative_derivative_below_01=float(np.mean(d < .1)),
                mean_squared_relative_derivative=float(np.mean(d ** 2)),
                severe_global_saturation=bool(np.mean(d < .01) >= .5))

results = {}
for mode in ('quantum', 'dephased'):
    rec = provenance['models'][mode]
    path = prior / rec['file']
    assert digest(path) == rec['sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)
    w, b, effect = (state[k].detach().numpy().astype(np.float64)
                    for k in ('prepare.weight', 'prepare.bias', 'effect'))
    assert w.shape == (12, 1028) and b.shape == (12,) and effect.shape == (40, 2, 3, 2)
    z = x @ w.T + b
    results[mode] = dict(prepare=stats(z), effect=stats(effect),
                         checkpoint_sha256=rec['sha256'], selected_epoch=rec['selected_epoch'])

fixture = np.array([-6., -3., -1., 0., 1., 3., 6.])
eps = 1e-5
numeric = (np.tanh(fixture + eps) - np.tanh(fixture - eps)) / (2 * eps)
e = np.exp(-2 * np.abs(fixture)); analytic = 4 * e / (1 + e) ** 2
residual = float(np.max(np.abs(analytic - numeric)))
assert residual < 1e-8
print(json.dumps(dict(status='COMPLETE', profile=cfg['profile'], sample_rows=len(indices),
    words=len(x), selection_id_hash_sha256=hashlib.sha256(''.join(meta[i]['id_hash'] for i in indices).encode()).hexdigest(),
    actual_selected_input_bytes_sha256=actual_inputs.hexdigest(), results=results,
    derivative_fixture_max_abs=residual, seconds=time.monotonic() - started,
    optimizer_updates=0, full_circuit_forward_calls=0, gpu_operations=0, test_rows_read=0,
    cos_writes=0, formal_training_started=False, limitations=cfg['limitations']), allow_nan=False))
