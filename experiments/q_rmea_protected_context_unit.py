"""Bounded CPU first-order checks using the preserved private two-case fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baselines.retacred.rmea import RMEAHead
from baselines.retacred.rmea_protected_context import (
    CUTOFF, KINDS, ProtectedContextHead, _SpanProjector)

FIXTURE_SHA = 'a241fb5d7e3e63e54210f045f72297d9dcbbbeb43771fe9932f1486a5264c1d8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, tolerance, name):
    error = float((a-b).abs().max())
    require(error <= tolerance, f'{name}: {error} > {tolerance}')
    return error


def projector_checks():
    generator = torch.Generator().manual_seed(88013)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.complex128)
    passed = {}
    # A*B keeps deficient examples on a constant-rank manifold during gradcheck.
    for rank in (2, 4, 8):
        a = torch.linalg.qr(rand(8, rank)).Q.detach().requires_grad_()
        b = torch.linalg.qr(rand(8, rank)).Q.mH.detach().requires_grad_()
        passed[f'rank{rank}_factor_manifold'] = torch.autograd.gradcheck(
            lambda x, y: _SpanProjector.apply(x@y)[0], (a, b), fast_mode=True,
            eps=1e-6, atol=3e-6, rtol=3e-5)
    # Full-column rank repeated nonzero singular values: arbitrary column
    # perturbations preserve rank, unlike arbitrary perturbations of padded zeros.
    a = torch.linalg.qr(rand(8, 3)).Q.detach().requires_grad_()
    passed['repeated_column_rank3'] = torch.autograd.gradcheck(
        lambda x: _SpanProjector.apply(torch.nn.functional.pad(x, (0, 5)))[0],
        (a,), fast_mode=True, eps=1e-6, atol=3e-6, rtol=3e-5)
    # A nonzero discarded singular value makes gap conditioning explicit.
    u, v = torch.linalg.qr(rand(8, 8)).Q, torch.linalg.qr(rand(8, 8)).Q
    z = (u @ torch.diag(torch.tensor([2., 1., .5, 1e-12, 0., 0., 0., 0.],
                                    dtype=torch.complex128)) @ v.mH).requires_grad_()
    projector, singular = _SpanProjector.apply(z)
    pgrad = torch.autograd.grad((projector.real.square().sum()), z)[0]
    require(torch.isfinite(pgrad).all(), 'Nonfinite truncated projector derivative')
    passed['finite_truncated_derivative_only'] = True
    passed['truncated_rank'] = int((singular > singular[0]*CUTOFF).sum())
    return passed


def batch(cases):
    length = max(c['h'].shape[1] for c in cases)
    h = torch.zeros(len(cases), length, 1024, dtype=torch.float64)
    roles = torch.zeros(len(cases), length, 4, dtype=torch.float64)
    mask = torch.zeros(len(cases), length, dtype=torch.bool)
    for i, c in enumerate(cases):
        n = c['h'].shape[1]
        h[i, :n], roles[i, :n], mask[i, :n] = c['h'][0], c['roles'][0], c['mask'][0]
    return h, roles, mask


def gradients(head, inputs, targets, micro):
    head.zero_grad(set_to_none=True)
    for start in range(0, len(targets), micro):
        logits, _ = head(*(x[start:start+micro] for x in inputs))
        (torch.nn.functional.cross_entropy(logits, targets[start:start+micro],
                                           reduction='sum')/len(targets)).backward()
    result = {}
    for name, parameter in head.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                f'Missing or nonfinite gradient: {name}')
        result[name] = parameter.grad.detach().clone()
    return result


def run(args):
    torch.set_num_threads(2)
    require(hashlib.sha256(args.fixture.read_bytes()).hexdigest() == FIXTURE_SHA,
            'Immutable source fixture mismatch')
    original = torch.load(args.fixture, map_location='cpu', weights_only=True)
    cases, state = original['cases'], original['state']
    require(len(cases) == 2, 'Expected frozen first negative and first positive')
    require(not any(args.output.glob('*')) if args.output.exists() else True,
            'Output must be new/empty')
    checks = projector_checks()
    inputs = batch(cases)
    # Targets only define a derivative loss, not utility screening or selection.
    targets = torch.tensor([0, 1])
    modes, replay = {}, []
    for kind in KINDS:
        head = ProtectedContextHead(kind).double().eval()
        head.load_state_dict(state, strict=True)
        require(sum(p.numel() for p in head.parameters()) == 81028,
                'Parameter count differs from native')
        require(list(head.state_dict()) == list(state), 'Checkpoint key drift')
        d = head(*inputs, details=True)
        whole = gradients(head, inputs, targets, 2)
        micro = gradients(head, inputs, targets, 1)
        error = max(near(whole[k], micro[k], 2e-8, f'{kind}/{k}') for k in whole)
        rows = []
        for i, case in enumerate(cases):
            with torch.no_grad():
                one = head(case['h'], case['roles'], case['mask'], details=True)
            n = case['h'].shape[1]
            near(d['logits'][i], one['logits'][0], 1e-10, 'Padding/batch logits')
            near(d['alpha'][i, :, :n], one['alpha'][0], 1e-10, 'Padding/batch alpha')
            require(float(d['alpha'][i, :, n:].abs().sum()) == 0., 'Nonzero pad attention')
            if kind == 'native':
                require(torch.equal(one['logits'][0], case['native_logits']), 'Native logits drift')
                require(torch.equal(one['alpha'][0], case['native_alpha']), 'Native attention drift')
            elif kind == 'protected_context':
                near(one['alpha'][0], case['matched']['alpha'], 1e-10, 'Frozen filter parity')
                near(one['logits'][0], case['logits'][1], 1e-10, 'Frozen logit parity')
            if kind != 'native':
                s = one['singular_values'][0]
                rank = int((s > s[0]*CUTOFF).sum())
                eigenvalues = s.square()
                gap = float(eigenvalues[rank-1]-eigenvalues[rank]) if 0 < rank < 8 else None
                rows.append(dict(rank=rank, cross_gap=gap,
                                 max_correction=float(one['correction'].abs().max())))
                # Check invariants for the actual model's fully differentiable solve.
                if kind != 'unprotected_context':
                    before = one['phi'].conj() @ one['psi'][0].T
                    after = one['corrected'][0].conj() @ one['psi'][0].T
                    residual = float((before[:, one['mention'][0]]-after[:, one['mention'][0]]).abs().max())
                    dropped = float(s[rank]) if rank < len(s) else 0.
                    bound = dropped*float(one['correction'][0].norm(dim=-1).max())+1e-9
                    require(residual <= bound, 'Protected amplitude bound')
                near(torch.linalg.eigvalsh(one['used_covariance'][0]),
                     torch.linalg.eigvalsh(one['covariance'][0]), 1e-10,
                     'Covariance spectral control')
            replay.append(dict(kind=kind, case=i, logits=one['logits'].detach(),
                               alpha=one['alpha'].detach()))
        # Central difference of the actual constrained model with respect to
        # prepare/effect parameters, not a detached-projector surrogate.
        directional = {}
        if kind != 'native':
            for name in ('prepare.weight', 'effect'):
                parameter = dict(head.named_parameters())[name]
                generator = torch.Generator().manual_seed(91037)
                direction = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                direction /= direction.norm()
                analytic = float((micro[name]*direction).sum())
                saved = parameter.detach().clone()
                losses = []
                with torch.no_grad():
                    for sign in (1., -1.):
                        parameter.copy_(saved+sign*1e-6*direction)
                        logits, _ = head(*inputs)
                        losses.append(float(torch.nn.functional.cross_entropy(logits, targets)))
                    parameter.copy_(saved)
                numeric = (losses[0]-losses[1])/2e-6
                residual = abs(analytic-numeric)
                require(residual <= 3e-6+3e-5*abs(numeric), f'{kind}/{name} directional derivative')
                directional[name] = dict(analytic=analytic, numeric=numeric, residual=residual)
        modes[kind] = dict(parameters=81028, micro_gradient_max_error=error,
                           finite_gradients=True, rows=rows, directional=directional)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = args.output/'private-replay.pt'
    torch.save(dict(state=state, cases=cases, outputs=replay), saved)
    restored = torch.load(saved, map_location='cpu', weights_only=True)
    for record in restored['outputs']:
        head = ProtectedContextHead(record['kind']).double().eval()
        head.load_state_dict(restored['state'], strict=True)
        case = restored['cases'][record['case']]
        with torch.no_grad():
            logits, alpha = head(case['h'], case['roles'], case['mask'])
        require(torch.equal(logits, record['logits']) and torch.equal(alpha, record['alpha']),
                'Fresh model saved fixture replay mismatch')
    summary = dict(schema='q-rmea.protected-context.gradient-unit.v1', status='complete',
                   projector_checks=checks, modes=modes, fixture_sha256=FIXTURE_SHA,
                   private_replay_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                   private_replay_bytes=saved.stat().st_size, replay='bitwise_pass',
                   optimizer_steps=0, device='cpu', derivative_scope='first_order_constant_rank_only',
                   sample_trace_v2=False, formal_ready=False, L5='NOT_ACCEPTED')
    data = (json.dumps(summary, indent=2, allow_nan=False)+'\n').encode()
    (args.output/'gradient-summary.json').write_bytes(data)
    (args.output/'RUN_COMPLETE').write_text(hashlib.sha256(data).hexdigest()+'\n')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
