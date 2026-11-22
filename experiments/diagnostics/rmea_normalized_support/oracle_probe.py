"""Finite normalized-effect support on fixed exposed TRAIN inputs; no fitting."""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'inputs', 'checkpoint', 'population', 'math_target', 'design', 'output', 'progress'):
        p.add_argument('--' + n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--attempt', required=True)
    a = p.parse_args()
    cfg = json.loads(a.design.read_bytes())
    signal.alarm(cfg['timeout_seconds'])
    sys.path.insert(0, str(a.math_target))
    import numpy as np
    import torch
    import cvxpy as cp
    import scipy, clarabel, scs, osqp
    versions = dict(numpy=np.__version__, torch=torch.__version__, cvxpy=cp.__version__,
                    scipy=scipy.__version__, clarabel=clarabel.__version__, scs=scs.__version__, osqp=osqp.__version__)
    for name, expected in cfg['math_versions'].items():
        assert versions[name] == expected, (name, versions[name])
    for n, expected in cfg['source_hashes'].items():
        assert sha(a.source / n) == expected
    assert sha(a.checkpoint) == cfg['checkpoint_sha256']
    assert sha(a.population) == cfg['population_sha256']
    assert sha(a.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    assert sha(Path(__file__)) == cfg['probe_sha256']
    population = json.loads(a.population.read_bytes())
    assert population['selected_count'] == 32 and population['train_total'] == 58465
    assert not (a.output / 'RUN_COMPLETE').exists()
    a.output.mkdir(parents=True, exist_ok=True)
    def save(path, value):
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2))
        pending.replace(path)
    def progress(phase, current):
        save(a.progress, dict(schema='research.progress.v1', attempt=a.attempt, phase=phase,
            current=current, total=32, unit='case', observed_at=time.time(), heartbeat_at=time.time()))
    sys.path.insert(0, str(a.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    head = RMEAHead('quantum').double().eval()
    head.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    cache = a.inputs / 'discovery/cache'
    meta = json.loads((cache / 'metadata.json').read_bytes())['train']
    data = {n:np.load(cache / ('train-' + n + '.npy'), mmap_mode='r', allow_pickle=False)
            for n in ('h', 'roles', 'offsets')}
    eye = np.eye(8, dtype=np.complex128)
    options = dict(solver='CLARABEL', max_iter=cfg['solver_max_iter'], tol_gap_abs=1e-10,
                   tol_gap_rel=1e-10, tol_feas=1e-10, verbose=False)
    rows = []
    progress('normalized_support', 0)
    for number, case in enumerate(population['cases']):
        started = time.perf_counter()
        i = case['index']
        assert meta[i]['id_hash'] == case['id_hash']
        lo, hi = map(int, data['offsets'][i:i+2])
        h = torch.from_numpy(np.array(data['h'][lo:hi], dtype=np.float64, copy=True))[None]
        roles = torch.from_numpy(np.array(data['roles'][lo:hi], dtype=np.float64, copy=True))[None]
        mask = torch.ones(1, hi-lo, dtype=torch.bool)
        with torch.no_grad():
            detail = head(h, roles, mask, details=True)
        psi = detail['psi'][0].numpy()
        phi = detail['phi'][0].numpy()
        u = detail['utilities'][0, 0].numpy()
        native_alpha = detail['alpha'][0, 0].numpy()
        mention = ((roles[0, :, 2] != 0) | (roles[0, :, 3] != 0)).numpy()
        assert int(mention.sum()) == case['mention_tokens']
        b = psi.T @ psi.conj() + len(psi) * EPS * eye
        eig, basis = np.linalg.eigh(b)
        assert eig.min() > 0
        invhalf = (basis * (1 / np.sqrt(eig))) @ basis.conj().T
        half = (basis * np.sqrt(eig)) @ basis.conj().T
        mapped = np.einsum('ij,tj->ti', invhalf, psi)
        ds = np.einsum('ti,tj->tij', mapped, mapped.conj()) + EPS * (invhalf @ invhalf)[None]
        native_rho = np.outer(phi, phi.conj())
        native_z = half @ native_rho @ half / np.trace(b @ native_rho).real
        c = np.einsum('t,tij->ij', u, ds)
        c = (c + c.conj().T) / 2
        fs = np.concatenate([eye[None] / np.sqrt(8), ds[mention]])
        norms = np.linalg.norm(fs, axis=(1,2))
        fs = fs / norms[:,None,None]
        rhs = np.concatenate([[1 / np.sqrt(8)], native_alpha[mention]]) / norms
        gram = np.einsum('aij,bij->ab', fs.conj(), fs).real
        gram = (gram + gram.T) / 2
        sv = np.linalg.svd(gram, compute_uv=False)
        row = dict(index=i, id_hash=case['id_hash'], mention_tokens=case['mention_tokens'],
                   finite=False, eigenvalue_b_min=float(eig.min()), constraint_singular_ratio=float(sv[-1]/sv[0]))
        def moments(z):
            return np.einsum('aij,ij->a', fs.conj(), z).real
        def repair(z):
            z = (z + z.conj().T) / 2
            residual = rhs - moments(z)
            coeff = np.linalg.lstsq(gram, residual, rcond=1e-12)[0]
            z = z + np.einsum('a,aij->ij', coeff, fs)
            return (z + z.conj().T) / 2
        def constraints(z):
            return [cp.real(cp.trace(f @ z)) == float(v) for f, v in zip(fs, rhs)]
        native_error = float(np.max(np.abs(moments(native_z) - rhs)))
        row['native_equality_error'] = native_error
        z = cp.Variable((8,8), hermitian=True)
        t = cp.Variable()
        centre_problem = cp.Problem(cp.Maximize(t), [z - t * eye >> 0, *constraints(z)])
        try:
            centre_problem.solve(**options)
            row['centre_status'] = centre_problem.status
            if centre_problem.status != cp.OPTIMAL or z.value is None:
                row['reason'] = 'No optimal centre under the frozen solver profile'
            else:
                centre = repair(z.value)
                sigma = float(np.linalg.eigvalsh(centre).min())
                row['centre_min_eigenvalue'] = sigma
                row['centre_equality_error'] = float(np.max(np.abs(moments(centre)-rhs)))
                if sigma <= cfg['centre_min_eigenvalue'] or row['centre_equality_error'] > 1e-12:
                    row['reason'] = 'Strict corrected feasible centre not resolved'
                else:
                    endpoints = []
                    for sign in (-1,1):
                        z = cp.Variable((8,8), hermitian=True)
                        eq = constraints(z)
                        problem = cp.Problem(cp.Maximize(cp.real(cp.trace(sign * c @ z))), [z >> 0, *eq])
                        problem.solve(**options)
                        ep = dict(sign=sign, status=problem.status, accepted=False)
                        if problem.status == cp.OPTIMAL and z.value is not None:
                            corrected = repair(z.value)
                            negative = max(0., -float(np.linalg.eigvalsh(corrected).min()))
                            blend = (negative + 1e-12) / (sigma + negative + 1e-12)
                            feasible = (1-blend) * corrected + blend * centre
                            density = invhalf @ feasible @ invhalf
                            density = (density + density.conj().T) / 2
                            density /= np.trace(density).real
                            raw = np.einsum('ti,ij,tj->t', psi.conj(), density, psi).real + EPS
                            alpha = raw / raw.sum()
                            lambdas = np.array([float(q.dual_value) for q in eq])
                            slack = np.einsum('a,aij->ij', lambdas, fs) - sign*c
                            slack = (slack + slack.conj().T) / 2
                            # trace(feasible)=1, so an identity shift yields a numerical weak-dual bound.
                            shift = max(0., -float(np.linalg.eigvalsh(slack).min())) + 1e-10
                            dual_upper = float(lambdas @ rhs) + shift
                            objective = float(sign * (u @ alpha))
                            gap = dual_upper - objective
                            ep.update(logit=float(u @ alpha), dual_upper_signed=dual_upper,
                                numerical_duality_gap=gap, primal_z_min_eigenvalue=float(np.linalg.eigvalsh(feasible).min()),
                                density_min_eigenvalue=float(np.linalg.eigvalsh(density).min()),
                                equality_error=float(np.max(np.abs(moments(feasible)-rhs))),
                                mention_error=float(np.max(np.abs(alpha[mention]-native_alpha[mention]))),
                                context_mass_error=abs(float(alpha[~mention].sum()-native_alpha[~mention].sum())),
                                psd_repair_blend=blend, numerical_density_rank=int((np.linalg.eigvalsh(density)>1e-7).sum()))
                            ep['accepted'] = bool(ep['equality_error']<=1e-12 and ep['mention_error']<=1e-9
                                and ep['context_mass_error']<=1e-9 and ep['primal_z_min_eigenvalue']>=-1e-12
                                and ep['density_min_eigenvalue']>=-1e-12 and -1e-8<=gap<=cfg['duality_gap_max'])
                        endpoints.append(ep)
                    row['endpoints'] = endpoints
                    row['finite'] = all(q['accepted'] for q in endpoints)
                    if row['finite']:
                        width = endpoints[1]['logit']-endpoints[0]['logit']
                        native_value = float(u @ native_alpha)
                        row.update(width=width, native_logit_without_bias=native_value,
                            unconstrained_width=float(np.linalg.eigvalsh(c)[-1]-np.linalg.eigvalsh(c)[0]),
                            native_inside=bool(endpoints[0]['logit']-1e-8<=native_value<=endpoints[1]['logit']+1e-8),
                            nonzero=bool(width>=cfg['width_min']))
        except cp.error.SolverError as exc:
            row['reason'] = 'Frozen solver failure: ' + str(exc)[:200]
        row['seconds'] = time.perf_counter()-started
        rows.append(row)
        save(a.output / 'partial.json', dict(rows=rows, versions=versions))
        progress('normalized_support', number+1)
    finite = [q for q in rows if q['finite']]
    positive = [q for q in finite if q['nonzero'] and q['native_inside']]
    gates = dict(all_cases_retained=len(rows)==32, finite_nonzero_coverage=len(positive)>=cfg['positive_required'],
                 all_native_feasible=all(q['native_equality_error']<=1e-10 for q in rows),
                 native_in_interval=all(q['native_inside'] for q in finite))
    summary = dict(status='complete', profile=cfg['profile'], design_sha256=sha(a.design),
        probe_sha256=sha(Path(__file__)), versions=versions, rows=rows, selected_count=32,
        finite_count=len(finite), positive_count=len(positive), gates=gates, local_gate=all(gates.values()),
        width_range=[min(q['width'] for q in finite), max(q['width'] for q in finite)] if finite else None,
        scope='Exposed TRAIN, relation0, unrestricted PSD state-port numerical feasibility to explicit tolerances; no pure-optimality comparison or shared learned implementation, task utility, novelty, independent confirmation or L5.',
        labels_used=False, optimizer_updates=0, cos_writes=0, L5='NOT_ACCEPTED')
    save(a.output/'summary.json', summary)
    save(a.output/'RUN_COMPLETE', dict(artifacts={'summary.json':sha(a.output/'summary.json')}))
    progress('complete',32)
    print(json.dumps({k:summary[k] for k in ('status','selected_count','finite_count','positive_count','width_range','gates','local_gate','L5')}), flush=True)

if __name__ == '__main__':
    main()
