"""Fixed synthetic falsifiers; execute only through the shared scheduler."""
import itertools
import math

import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
from experiments.rmea_utility_measurement.cross_fold import KINDS, CrossFoldUtilityHead, fold_attention

CHECK_NAMES = [
    'native_eta0_short_sequences_valid_order_and_padding',
    'all40_invalid_interfaces_and_parameter_isolation',
    'recipient_dependency_and_fold_constant_shift',
    'signed_cross_response_and_nonnegative_self_response',
    'zero_utility_jacobian_signed_pairing_and_self_gram',
    'independent_noise_identity_and_correlated_failure_witnesses',
    'end_to_end_gradients_and_utility_directional_difference',
    'deterministic_state_reload_and_kind_rejection',
]


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261120)
    checks = []
    def record(**values):
        checks.append(dict(name=CHECK_NAMES[len(checks)], status='PASS', **values))
    def close(a, b, tol=2e-9):
        assert torch.allclose(a, b, atol=tol, rtol=0), float((a-b).abs().max())
    def states(shape):
        x = torch.randn(shape, dtype=torch.float64) + 1j*torch.randn(shape, dtype=torch.float64)
        return x / x.norm(dim=-1, keepdim=True)
    psi, phi = states((1, 7, 8)), states((40, 8))
    mask = torch.tensor([[True, False, True, True, False, True, True]])
    u = torch.randn((1, 40, 7), dtype=torch.float64) * .3
    dirty_psi, dirty_u = psi.clone(), u.clone()
    dirty_psi[~mask] = complex(float('nan'), 0); dirty_u.masked_fill_(~mask[:, None], float('nan'))
    for kind in KINDS:
        d = fold_attention(dirty_psi, phi, dirty_u, mask, kind=kind, eta=0.)
        close(d['alpha'], d['native_alpha'])
        assert torch.equal(d['fold_masks'][0], torch.tensor([[True, False, False, True, False, False, True]]))
        assert torch.equal(d['alpha'].masked_select(~mask[:, None]), torch.zeros(80, dtype=torch.float64))
        close(d['alpha'].sum(-1), torch.ones((1, 40), dtype=torch.float64))
        for n in ((1, 2, 3) if kind == 'cross_fold' else (1, 2)):
            m = torch.zeros_like(mask); m[0, torch.where(mask[0])[0][:n]] = True
            short = fold_attention(psi, phi, u, m, kind=kind)
            close(short['alpha'], short['native_alpha'])
    record(masked_nan_ignored=True, all40_classes=True, short_native=True)

    invalid = [(psi, phi[1:], u, mask), (psi, phi, u.float(), mask),
               (psi, phi, u, mask.float()), (psi, phi, u, torch.zeros_like(mask)),
               (psi*2, phi, u, mask)]
    bad = u.clone(); bad[0, 0, 0] = float('nan'); invalid.append((psi, phi, bad, mask))
    for args in invalid:
        try: fold_attention(*args)
        except ValueError: pass
        else: raise AssertionError('Invalid interface accepted')
    original = RMEAHead('quantum').double()
    initial = {k: v.clone() for k, v in original.state_dict().items()}
    for kind in KINDS:
        head = CrossFoldUtilityHead(kind).double(); head.load_state_dict(initial)
        assert set(head.state_dict()) == set(initial)
        assert sum(p.numel() for p in head.parameters()) == 81028
    record(rejected=len(invalid), stored_parameters=81028)

    base = fold_attention(psi, phi, u, mask)
    fa, fb = base['fold_masks']
    changed = u.clone(); changed[:, :, fa[0]] += torch.linspace(-2., 2., int(fa.sum()))
    perturbed = fold_attention(psi, phi, changed, mask)
    close(base['alpha'][:, :, fa[0]], perturbed['alpha'][:, :, fa[0]], 0.)
    assert float((base['alpha'][:, :, fb[0]]-perturbed['alpha'][:, :, fb[0]]).abs().max()) > 1e-8
    shifts = 1.3*fa[:, None].to(u.dtype) - .7*fb[:, None].to(u.dtype)
    for kind in KINDS:
        before = fold_attention(psi, phi, u, mask, kind=kind)
        after = fold_attention(psi, phi, u+shifts, mask, kind=kind)
        close(before['alpha'], after['alpha'])
        masses = before['samples'][0]['native_fold_masses']
        close(after['aggregate']-before['aggregate'], (1.3*masses[0]-.7*masses[1])[None])
    record(own_recipient_routing_independent=True, opposite_recipient_changes=True, shift_dtype=str(shifts.dtype))

    # Two identical state pairs; sign of inter-fold utility alignment is controlled.
    sp = torch.zeros((1, 4, 8), dtype=torch.complex128)
    sp[0, :2, 0] = 1.; sp[0, 2:, 1] = 1.
    ph = torch.zeros((40, 8), dtype=torch.complex128); ph[:, :2] = 2**-.5
    mm = torch.ones((1, 4), dtype=torch.bool)
    def utility(vector): return torch.tensor(vector, dtype=torch.float64)[None, None].expand(1, 40, 4).clone()
    aligned, opposing = utility([1., 1., -1., -1.]), utility([1., -1., -1., 1.])
    step = 1e-5
    # The public contract eta>=0 permits a right derivative at native eta0.
    slopes = {}
    for label, values in [('aligned', aligned), ('opposing', opposing)]:
        for kind in KINDS:
            zero = fold_attention(sp, ph, values, mm, kind=kind, eta=0.)
            plus = fold_attention(sp, ph, values, mm, kind=kind, eta=step)
            slopes[label+'_'+kind] = float(((plus['aggregate']-zero['aggregate'])/step)[0, 0])
            left, right = zero['samples'][0]['donor_updates']
            aa, ab = left['covariance'][0], right['covariance'][0]
            na, nb = left['scale'][0], right['scale'][0]
            p = ph[..., :, None]*ph.conj()[..., None, :]
            q = torch.eye(8, dtype=torch.complex128)-p
            lp, lm = (1+EPS)/(1+8*EPS), EPS/(1+8*EPS)
            ell = 1/((1+8*EPS)*math.log((1+EPS)/EPS))
            def mean_map(a): return lp*(p@a@p)+lm*(q@a@q)+ell*(p@a@q+q@a@p)
            total = (torch.einsum('rd,btd->brt', ph.conj(), sp).abs().square()+EPS).sum(-1)[0]/(1+8*EPS)
            if kind == 'cross_fold':
                theory = torch.einsum('rij,rji->r', aa, mean_map(ab)).real*(1/na+1/nb)/total
            else:
                theory = (torch.einsum('rij,rji->r', aa, mean_map(aa)).real/na
                          +torch.einsum('rij,rji->r', ab, mean_map(ab)).real/nb)/total
            close(((plus['aggregate']-zero['aggregate'])/step)[0], theory, 1e-5)
    assert slopes['aligned_cross_fold'] > 0 and slopes['opposing_cross_fold'] < 0
    assert slopes['aligned_self_fold'] > 0 and slopes['opposing_self_fold'] > 0
    record(slopes=slopes, scope='Synthetic right derivatives; no CE theorem')

    def alpha_for(v, kind):
        return fold_attention(sp, ph, v[None, None].expand(1, 40, 4), mm, kind=kind)['alpha'][0, 0]
    zero = torch.zeros(4, dtype=torch.float64)
    jc = torch.autograd.functional.jacobian(lambda v: alpha_for(v, 'cross_fold'), zero)
    js = torch.autograd.functional.jacobian(lambda v: alpha_for(v, 'self_fold'), zero)
    a, b = torch.tensor([0, 2]), torch.tensor([1, 3])
    close(jc[a[:, None], a], torch.zeros((2, 2), dtype=torch.float64))
    close(jc[b[:, None], b], torch.zeros((2, 2), dtype=torch.float64))
    close(jc, jc.T); close(js, js.T)
    ec, es = torch.linalg.eigvalsh(jc), torch.linalg.eigvalsh(js)
    close(ec, -ec.flip(0)); assert ec[0] < -1e-8 and ec[-1] > 1e-8 and es.min() > -2e-9
    direction = torch.tensor([.2, -.7, .4, .1], dtype=torch.float64)
    fd = (alpha_for(step*direction, 'cross_fold')-alpha_for(-step*direction, 'cross_fold'))/(2*step)
    close(jc@direction, fd, 2e-7)
    record(cross_eigenvalues=ec.tolist(), self_eigenvalues=es.tolist(), jacobian_fd_error=float((jc@direction-fd).abs().max()))

    def noise_mean(vectors, kind):
        values = torch.tensor(vectors, dtype=torch.float64)[:, None].expand(-1, 40, 4)*.2
        count = len(vectors)
        return float(fold_attention(sp.expand(count, -1, -1), ph, values,
            mm.expand(count, -1), kind=kind)['aggregate'][:, 0].mean())
    independent = list(itertools.product((-1., 1.), repeat=4))
    positive = [[x, x, y, y] for x, y in itertools.product((-1., 1.), repeat=2)]
    negative = [[x, -x, y, -y] for x, y in itertools.product((-1., 1.), repeat=2)]
    means = dict(independent_cross=noise_mean(independent, 'cross_fold'),
                 independent_self=noise_mean(independent, 'self_fold'),
                 correlated_cross=noise_mean(positive, 'cross_fold'),
                 anticorrelated_cross=noise_mean(negative, 'cross_fold'))
    assert abs(means['independent_cross']) < 2e-10 and means['independent_self'] > 1e-8
    assert means['correlated_cross'] > 1e-8 and means['anticorrelated_cross'] < -1e-8
    record(means=means, scope='Exhaustive bounded synthetic signs; not an NLP noise model')

    h = torch.randn((1, 4, 1024), dtype=torch.float64)*.1
    roles = torch.zeros((1, 4, 4), dtype=torch.float64); roles[0, 0, 2]=1.; roles[0, 3, 3]=1.
    for kind in KINDS:
        head = CrossFoldUtilityHead(kind).double(); head.load_state_dict(initial)
        z, _ = head(h, roles, mm); F.cross_entropy(z, torch.tensor([7])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    uv = u.clone().requires_grad_(); weights = torch.linspace(-.4, .7, 40, dtype=torch.float64)[None]
    delta = torch.randn_like(uv)*.1
    def objective(v): return (fold_attention(psi, phi, v, mask)['aggregate']*weights).sum()
    grad = torch.autograd.grad(objective(uv), uv)[0]
    fd = (objective(uv+step*delta)-objective(uv-step*delta))/(2*step)
    close((grad*delta).sum(), fd, 2e-6)
    state_leaf = psi.clone().requires_grad_()
    state_direction = states(psi.shape)*.1
    def state_objective(value):
        normalized = value/value.norm(dim=-1, keepdim=True)
        return (fold_attention(normalized, phi, u, mask)['aggregate']*weights).sum()
    state_grad = torch.autograd.grad(state_objective(state_leaf), state_leaf)[0]
    state_fd = (state_objective(state_leaf+step*state_direction)
                -state_objective(state_leaf-step*state_direction))/(2*step)
    state_ad = (state_grad.conj()*state_direction).sum().real
    close(state_ad, state_fd, 2e-6)
    record(utility_fd_error=float(((grad*delta).sum()-fd).abs()),
           state_and_mass_fd_error=float((state_ad-state_fd).abs()), full_parameter_gradients=True)

    for kind in KINDS:
        first = CrossFoldUtilityHead(kind).double(); first.load_state_dict(initial)
        second = CrossFoldUtilityHead(kind).double(); second.load_state_dict(first.state_dict())
        assert all(torch.equal(x, y) for x, y in zip(first(h, roles, mm), second(h, roles, mm)))
        parity = CrossFoldUtilityHead(kind, eta=0.).double(); parity.load_state_dict(initial)
        za, aa = parity(h, roles, mm); zb, ab = original(h, roles, mm)
        close(za, zb); close(aa, ab)
    for kind in ('full', 'unknown'):
        try: CrossFoldUtilityHead(kind)
        except ValueError: pass
        else: raise AssertionError('Undeclared kind accepted')
    record(source_native_parity=True, state_reload=True)
    assert len(checks) == len(CHECK_NAMES)
    return dict(status='CROSS_FOLD_NUMERICAL_CANARY_COMPLETE', checks=checks,
        check_count=len(checks), seed=20261120, task_training=False, optimizer_steps=0,
        real_dataset_reads=0, new_checkpoints=0, COS_requests=0, cloud_writes=0,
        independent_confirmation=False, formal_handoff=False, L5='NOT_ACCEPTED',
        qualification='Synthetic implementation checks only; no identified task-noise, novelty, task utility or trace-readiness claim.')
