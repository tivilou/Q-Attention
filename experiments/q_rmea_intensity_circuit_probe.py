"""Frozen zero-fit circuit feasibility diagnostic on the previous TRAIN32 cohort."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes((json.dumps(value, indent=2) + '\n').encode())
    temp.replace(path)


def measure(detail, roles, bias, circuit, eps):
    psi, phi = detail['psi'][0].detach(), detail['phi'][0].detach()
    theta = detail['effect_angles'][0].detach().reshape(12)
    mention = (roles[0, :, 2] != 0) | (roles[0, :, 3] != 0)
    selected, contexts = psi[mention], psi[~mention]
    m = len(selected)
    row = dict(mention_tokens=m, contexts=len(contexts), status='inapplicable', positive=False)
    if not len(contexts) or m > 11:
        row['reason'] = 'outside_frozen_full_row_rank_nonzero_kernel_domain'
        return row
    covariance = contexts.T @ contexts.conj() / len(contexts)
    utilities = detail['utilities'][0, 0].detach()

    def quantities(angles):
        state = circuit(angles.reshape(2, 3, 2))
        intensity = (psi.conj() @ state).abs().square()
        raw = intensity + eps
        logit = (raw / raw.sum() * utilities).sum() + bias
        energy = (state.conj() @ (covariance @ state)).real + .125 * (state - phi).abs().square().sum()
        return torch.cat((intensity[mention], energy[None], logit[None], state.real, state.imag))

    baseline = quantities(theta).detach()
    assert float((circuit(theta.reshape(2, 3, 2)) - phi).abs().max()) < 1e-12
    assert abs(float(baseline[m + 1] - detail['logits'][0, 0])) < 1e-12
    jac = torch.autograd.functional.jacobian(quantities, theta).detach()
    j, ge, gl = jac[:m], jac[m], jac[m + 1]
    state_jac = torch.complex(jac[m + 2:m + 10], jac[m + 10:m + 18])
    horizontal = state_jac - phi[:, None] * (phi.conj() @ state_jac)[None]
    physical_singular = torch.linalg.svdvals(torch.cat((horizontal.real, horizontal.imag), dim=0))
    row['physical_jacobian_rank_at_1e8'] = int((physical_singular > 1e-8 * physical_singular[0]).sum())
    u, s, vh = torch.linalg.svd(j, full_matrices=False)
    row['constraint_singular_ratio'] = float(s[-1] / s[0]) if s[0] > 0 else 0.
    if row['constraint_singular_ratio'] < 1e-6:
        row['reason'] = 'outside_well_conditioned_constraint_branch'
        return row
    b = (vh.T / s) @ u.T
    projector = torch.eye(12, dtype=torch.float64) - b @ j
    ae, al = projector @ ge, projector @ gl
    row['projected_energy_ratio'] = float(ae.norm() / ge.norm()) if ge.norm() > 0 else 0.
    row['projected_logit_ratio'] = float(al.norm() / gl.norm()) if gl.norm() > 0 else 0.
    if min(row['projected_energy_ratio'], row['projected_logit_ratio']) < 1e-8:
        row['reason'] = 'unresolved_energy_or_task_tangent'
        return row
    ae, al = ae / ae.norm(), al / al.norm()
    tangent = -ae - (1 if ae @ al >= 0 else -1) * al
    tangent = tangent / tangent.norm()
    assert float((j @ tangent).abs().max()) < 1e-8 and float(ge @ tangent) < 0 and abs(float(gl @ tangent)) > 0
    # Fixed step and fixed-Jacobian chord correction; no convergence theorem or
    # adaptive rescue is assumed. Final residual is part of the frozen gate.
    step = .001
    normal = torch.zeros(12, dtype=torch.float64)
    with torch.no_grad():
        plain = quantities(theta + step * tangent)
        for _ in range(36):
            residual = quantities(theta + step * tangent + normal)[:m] - baseline[:m]
            normal = normal - b @ residual
        angles = theta + step * tangent + normal
        candidate = quantities(angles)
        vector = circuit(angles.reshape(2, 3, 2))
        error = max(float((candidate[:m] - baseline[:m]).abs().max()), abs(float(vector.abs().square().sum()) - 1))
        delta_e, delta_l = float(candidate[m] - baseline[m]), float(candidate[m + 1] - baseline[m + 1])
        in_range = bool((angles.abs() < math.pi).all())
        # If admitted, this is also representable by the source's raw effect
        # parameters: theta=pi*tanh(raw), with no change to its circuit family.
        raw_inverse_error = float((math.pi * torch.tanh(torch.atanh(angles / math.pi)) - angles).abs().max()) if in_range else None
    row.update(status='evaluated', reason='fixed_chord_result', step=step, iterations=36,
               constraint_residual=error, tangent_only_residual=float((plain[:m] - baseline[:m]).abs().max()),
               energy_delta=delta_e, relative_logit_delta=delta_l, angle_delta_norm=float((angles - theta).norm()),
               in_source_parameter_range=in_range, raw_inverse_error=raw_inverse_error,
               positive=error <= 1e-12 and delta_e < -1e-10 and abs(delta_l) >= 1e-6 and in_range)
    assert all(np.isfinite(row[k]) for k in ('constraint_residual', 'energy_delta', 'relative_logit_delta', 'angle_delta_norm'))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'baseline-run', 'pins', 'design', 'population', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.design.read_bytes())
    assert cfg['profile'] == 'born-intensity-circuit-train32-v1' and cfg['class_index'] == 0
    assert digest(args.population) == cfg['prior_population_sha256']
    population = json.loads(args.population.read_bytes())
    assert len(population['cases']) == 32 and population['train_total'] == 58465
    assert not (args.output / 'RUN_COMPLETE').exists(), 'Preserve completed output'
    for name, value in cfg['source_hashes'].items():
        assert digest(args.source_root / name) == value
    sys.path.insert(0, str(args.source_root))
    from baselines.retacred.rmea import RMEAHead, EPS, circuit
    assert EPS == 1e-6
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache = args.baseline_run / 'cache'
    for name, value in json.loads(args.pins.read_bytes()).items():
        path = cache / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == value['bytes'] and digest(path) == value['sha256']
    checkpoint = args.baseline_run / 'checkpoints/native-final.pt'
    assert digest(checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.baseline_run / 'manifest.json') == cfg['native_manifest_sha256']
    assert json.loads((args.baseline_run / 'manifest.json').read_bytes())['config']['dtype'] == 'float64/complex128'
    data = {k: np.load(cache / ('train-' + k + '.npy'), mmap_mode='r', allow_pickle=False) for k in ('h', 'roles', 'offsets')}
    metadata = json.loads((cache / 'metadata.json').read_bytes())['train']
    head = RMEAHead('quantum').double().eval()
    head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    rows = []
    for case in population['cases']:
        i = case['index']
        assert metadata[i]['id_hash'] == case['id_hash']
        lo, hi = map(int, data['offsets'][i:i + 2])
        inputs = [torch.from_numpy(np.array(data[k][lo:hi], dtype=np.float64, copy=True))[None] for k in ('h', 'roles')]
        with torch.no_grad():
            detail = head(*inputs, torch.ones(1, hi - lo, dtype=torch.bool), details=True)
        row = measure(detail, inputs[1], head.bias[0].detach(), circuit, EPS)
        assert row['mention_tokens'] == case['mention_tokens']
        rows.append(dict(index=i, id_hash=case['id_hash'], **row))
    summary = dict(profile=cfg['profile'], status='complete', design_sha256=digest(args.design),
                   population_sha256=digest(args.population), selected_count=32,
                   evaluated_count=sum(r['status'] == 'evaluated' for r in rows),
                   positive_count=sum(r['positive'] for r in rows), existence_gate=any(r['positive'] for r in rows),
                   rows=rows, labels_used=False, optimizer_updates=0, cos_operations=0,
                   claim_scope=cfg['claim_scope'], L5='NOT_ACCEPTED')
    emit(args.output / 'summary.json', summary)
    emit(args.output / 'RUN_COMPLETE', dict(artifacts={'summary.json': digest(args.output / 'summary.json')}))
    print(json.dumps({k: summary[k] for k in ('status', 'selected_count', 'evaluated_count', 'positive_count', 'existence_gate', 'L5')}))


if __name__ == '__main__':
    main()
