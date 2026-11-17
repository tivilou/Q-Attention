"""Source-bound CPU unit/preflight/diagnostic job; never trains a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from core import analyze, tv_rows
from legacy.rmea import RMEAHead, circuit

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def independent_circuit(angles):
    state = np.eye(8, dtype=np.complex128)[:, 0]
    for layer in range(2):
        for wire in range(3):
            y, z = angles[layer, wire]
            c, s = np.cos(y / 2), np.sin(y / 2)
            gate = np.diag(np.exp(np.array([-.5j, .5j]) * z)) @ np.array([[c, -s], [s, c]])
            full = np.ones((1, 1), np.complex128)
            for j in range(3):
                full = np.kron(full, gate if j == wire else np.eye(2))
            state = full @ state
        for control, target in [(0, 1), (1, 2), (2, 0)]:
            full = np.zeros((8, 8), np.complex128)
            for column in range(8):
                row = column ^ ((1 << (2-target)) if ((column >> (2-control)) & 1) else 0)
                full[row, column] = 1
            state = full @ state
    return state


def source_check(contract):
    for name, expected in contract['legacy_sources'].items():
        assert sha(HERE / 'legacy' / name) == expected


def unit():
    rng = np.random.default_rng(20261117)
    angles = rng.normal(size=(7, 2, 3, 2))
    expected = np.stack([independent_circuit(x) for x in angles])
    actual = circuit(torch.from_numpy(angles)).numpy()
    error = float(abs(expected-actual).max())
    assert error < 2e-14
    torch.manual_seed(20261117)
    head = RMEAHead('quantum').eval()
    h = torch.from_numpy(rng.normal(size=(2, 6, 1024)).astype(np.float32))
    roles = torch.from_numpy(rng.normal(size=(2, 6, 4)).astype(np.float32))
    h[0, 1] = h[0, 0]; roles[0, 1] = roles[0, 0]
    mask = torch.tensor([[True, True, True, False, False, False], [True]*6])
    results = []
    with torch.inference_mode():
        detail = head(h, roles, mask, details=True)
        for i in range(2):
            valid = mask[i].numpy()
            r = analyze(detail['phi'].numpy(), detail['psi'][i].numpy()[valid],
                        detail['alpha'][i].numpy()[:, valid], detail['utilities'][i].numpy()[:, valid],
                        head.bias.numpy(), detail['logits'][i].numpy(), i)
            results.append(r)
        trimmed = head(h[:1, :3], roles[:1, :3], mask[:1, :3], details=True)
        assert torch.equal(detail['alpha'][0, :, :3], trimmed['alpha'][0])
        assert torch.allclose(detail['logits'][0], trimmed['logits'][0], rtol=1e-6, atol=1e-6)
        assert torch.equal(detail['alpha'][0, :, 0], detail['alpha'][0, :, 1])
        assert torch.count_nonzero(detail['alpha'][0, :, 3:]) == 0
    # Constant signed utilities expose the omitted native row-mass correction.
    a = np.ones((40, 2)) / 2; a[1] *= 1 + 1e-7
    u = np.full((40, 2), -9.)
    p = np.zeros((40, 8), np.complex128); p[:, 0] = 1
    k = np.zeros((2, 8), np.complex128); k[:, 0] = 1
    native = (a*u).sum(-1)
    mass = analyze(p, k, a, u, np.zeros(40), native, 0)
    assert mass['mass_correction_max'] > 8.9e-7 and mass['logit_violation'] < 1e-12
    bias = np.zeros(40); bias[7] = 3
    stable = analyze(p, k, a, u, bias, native+bias, 0)
    assert stable['decision_certified'] and stable['changed_argmax'] == 7
    # Same almost-overlapping queries, zero TV or near-unit TV by context.
    x, y = 9999/10001, 200/10001
    q = np.zeros((40, 8), complex); q[:, 0] = x; q[:, 1] = y; q[1, 1] = -y
    values = []
    for states in [np.array([[1, 0], [1, 0]]), np.array([[y, x], [y, -x]])]:
        keys = np.zeros((2, 8), complex); keys[:, :2] = states
        alpha = abs(q.conj() @ keys.T)**2 + 1e-6
        alpha /= alpha.sum(-1, keepdims=True)
        values.append(float(tv_rows(alpha)[0, 1]))
    assert values[0] == 0 and values[1] > .998
    return dict(status='PASS', cases=['independent_dense_gate_reference', 'native_float32_full40_bounds',
                'padded_mask', 'duplicate_tokens', 'common_noop_reduction', 'signed_native_mass_correction',
                'context_counterexample'], dense_circuit_max_error=error,
                max_logit_violation=max(x['logit_violation'] for x in results),
                max_geometry_violation=max(x['geometry_violation'] for x in results),
                context_tv=values, model_training=False, research_data_read=False)


def inputs(contract, directory):
    for key, row in contract['inputs'].items():
        path = directory / row['name']
        assert path.is_file() and sha(path) == row['sha256'], 'Pinned input missing or mismatched: '+key
    head = RMEAHead('quantum').eval()
    payload = torch.load(directory / contract['inputs']['head']['name'], map_location='cpu', weights_only=True)
    head.load_state_dict(payload, strict=True)
    anchors = []
    with torch.inference_mode():
        for name in ['anchor0', 'anchor1']:
            with np.load(directory/contract['inputs'][name]['name'], allow_pickle=False) as data:
                h, r, m = [torch.from_numpy(data[k].copy()) for k in ['h', 'roles', 'mask']]
                logits, _ = head(h, r, m)
                expected = data['logits'].copy()
            assert np.allclose(logits.numpy(), expected, rtol=3e-4, atol=5e-5)
            anchors.append(float(abs(logits.numpy()-expected).max()))
    # Open train arrays only: validation/test outcomes are not used by this diagnostic.
    with np.load(directory / contract['inputs']['cache']['name'], allow_pickle=False) as data:
        assert not any(k.startswith('test') for k in data.files)
        train = {k:data['train_'+k].copy() for k in ['features', 'roles', 'offsets', 'targets', 'ids']}
    assert len(train['targets']) == 2048 and len(train['offsets']) == 2049
    assert train['offsets'][0] == 0 and train['offsets'][-1] == len(train['features'])
    assert (np.diff(train['offsets']) > 0).all()
    assert train['features'].dtype == np.float16 and train['features'].shape[1] == 1024
    assert train['roles'].dtype == np.float32 and train['roles'].shape == (len(train['features']), 4)
    assert train['targets'].dtype.kind in 'iu' and train['targets'].min() >= 0 and train['targets'].max() < 40
    assert hashlib.sha256(b''.join(train['ids'])).hexdigest() == contract['train_ids_sha256']
    assert len(set(train['ids'])) == 2048
    return head, train, anchors


def diagnose(head, data):
    n = len(data['targets']); counts = np.bincount(data['targets'], minlength=40)
    confusion = np.zeros((40, 40), np.int64)
    sums = {key:np.zeros((40, 40)) for key in ['raw_fidelity', 'effective_fidelity', 'actual_tv', 'ideal_tv']}
    maxima = {key:0. for key in ['geometry_violation', 'logit_violation', 'state_norm_residual',
              'ideal_native_tv_residual', 'native_mass_residual', 'mass_correction_max', 'reduction_residual']}
    total = {key:0. for key in ['baseline_ce', 'changed_ce', 'ce_upper', 'cyclic_logit_absolute_change_mean', 'cyclic_logit_bound_mean']}
    transitions = dict(corrected=0, harmed=0, prediction_changed=0, decision_certified=0, native_argmax_disagreements=0)
    with torch.inference_mode():
        for i, target in enumerate(data['targets']):
            lo, hi = map(int, data['offsets'][i:i+2])
            h = torch.from_numpy(data['features'][lo:hi].astype(np.float32))[None]
            roles = torch.from_numpy(data['roles'][lo:hi])[None]
            d = head(h, roles, torch.ones((1, hi-lo), dtype=torch.bool), details=True)
            r = analyze(d['phi'].numpy(), d['psi'][0].numpy(), d['alpha'][0].numpy(),
                        d['utilities'][0].numpy(), head.bias.numpy(), d['logits'][0].numpy(), int(target))
            for key in sums: sums[key] += r[key]
            for key in maxima: maxima[key] = max(maxima[key], r[key])
            for key in total: total[key] += r[key]
            original, changed = r['baseline_argmax'], r['changed_argmax']
            confusion[int(target), original] += 1
            transitions['corrected'] += int(original != target and changed == target)
            transitions['harmed'] += int(original == target and changed != target)
            transitions['prediction_changed'] += int(original != changed)
            transitions['decision_certified'] += int(r['decision_certified'])
            transitions['native_argmax_disagreements'] += int(r['native_argmax'] != original)
    pairs = [dict(labels=[i,j], **{key:float(value[i,j]/n) for key,value in sums.items()}) for i in range(40) for j in range(i+1,40)]
    assert len(pairs) == 780 and int(confusion.sum()) == n
    return dict(status='PASS_DESCRIPTIVE_ONLY', train_entries=n, class_counts=counts.tolist(),
                missing_support_classes=np.where(counts==0)[0].tolist(), confusion=confusion.tolist(),
                all_label_pairs=pairs, maxima=maxima, means={k:v/n for k,v in total.items()},
                transitions=transitions, intervention='query row r -> (r+1) mod40, fixed signed utilities/bias',
                inference='No p-values, utility/generalization/necessity/L5 or new-method claim.')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--action', choices=['unit','preflight','diagnose'], required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--inputs', type=Path)
    args = p.parse_args(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    contract = json.loads((HERE/'contract.json').read_bytes()); source_check(contract)
    assert not args.output.exists() or not any(args.output.iterdir()), 'Output must be new/empty'
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'unit': result = unit()
    else:
        assert args.inputs is not None
        head, data, anchors = inputs(contract, args.inputs)
        result = dict(status='PASS_INPUT_PREFLIGHT', anchors_max_error=anchors)
        if args.action == 'diagnose': result.update(diagnose(head, data))
    result.update(action=args.action, contract_sha256=sha(HERE/'contract.json'),
                  environment=dict(torch=torch.__version__, numpy=np.__version__), formal_training=False)
    save(args.output/'compact-report.json', result)
    save(args.output/'RUN_COMPLETE', dict(action=args.action, files={'compact-report.json':sha(args.output/'compact-report.json')}))
    print(json.dumps(dict(action=args.action, status=result['status'])))


if __name__ == '__main__':
    main()
