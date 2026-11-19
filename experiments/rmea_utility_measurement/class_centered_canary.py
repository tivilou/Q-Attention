"""Synthetic mechanism checks, only through the reviewed shared CPU recipe."""
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
from experiments.rmea_utility_measurement.head import UtilityMeasurementHead, measured_attention
from experiments.rmea_utility_measurement.class_centered import (
    KINDS, ClassCenteredUtilityHead, class_centered_measurement, class_centered_utilities)

CHECK_NAMES = [
    'native_and_full_source_parity_and_parameter_isolation',
    'all40_null_included_padding_and_invalid_inputs',
    'common_decoder_translation_and_construction_only_limit',
    'end_to_end_gradients_and_centered_decoder_null_direction',
    'utility_gradient_finite_difference_and_no_hidden_detach',
    'source_epsilon_helpful_and_harmful_label_witness',
    'zero_common_utility_and_single_token_gradients',
    'deterministic_state_reload_and_explicit_kind_rejection',
]


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261119)
    checks = []
    def record(name, **values):
        assert name == CHECK_NAMES[len(checks)]
        checks.append(dict(name=name, status='PASS', **values))
    def close(a, b, tol=2e-10):
        assert torch.allclose(a, b, atol=tol, rtol=0), float((a-b).abs().max())
    def states(shape):
        x = torch.randn(shape, dtype=torch.float64) + 1j*torch.randn(shape, dtype=torch.float64)
        return x / x.norm(dim=-1, keepdim=True)

    source = RMEAHead('quantum').double()
    initial = {n: v.clone() for n, v in source.state_dict().items()}
    h = torch.randn((1, 3, 1024), dtype=torch.float64) * .1
    roles = torch.zeros((1, 3, 4), dtype=torch.float64)
    roles[0, 0, 2] = 1.; roles[0, 1, 3] = 1.
    mask = torch.tensor([[True, True, False]])
    heads = {}
    for kind in KINDS:
        model = ClassCenteredUtilityHead(kind).double()
        model.load_state_dict(initial)
        assert set(model.state_dict()) == set(initial)
        assert sum(p.numel() for p in model.parameters()) == 81028
        heads[kind] = model
    old_full = UtilityMeasurementHead().double(); old_full.load_state_dict(initial)
    for kind, old in [('native', source), ('full', old_full)]:
        za, aa = heads[kind](h, roles, mask); zb, ab = old(h, roles, mask)
        assert torch.equal(za, zb) and torch.equal(aa, ab)
    assert all(torch.equal(initial[n], v) for n, v in source.state_dict().items())
    record(CHECK_NAMES[0], stored_parameters=81028)

    utility = torch.zeros((1, 40, 3), dtype=torch.float64)
    utility[:, 0] = 40.; utility[:, :, 2] = float('nan')
    centered, common = class_centered_utilities(utility, mask)
    close(common[0, 0, :2], torch.ones(2, dtype=torch.float64))
    close(centered[0, 0, :2], torch.full((2,), 39., dtype=torch.float64))
    close(centered[0, 1:, :2], torch.full((39, 2), -1., dtype=torch.float64))
    assert torch.equal(centered[:, :, 2], torch.zeros((1, 40), dtype=torch.float64))
    bad = utility.clone(); bad[0, 0, 0] = float('nan')
    invalid = [(utility[:, 1:], mask), (utility.float(), mask),
               (utility, torch.zeros_like(mask)), (bad, mask), (utility, mask.float())]
    for u, m in invalid:
        try: class_centered_utilities(u, m)
        except ValueError: pass
        else: raise AssertionError('Invalid utility interface accepted')
    record(CHECK_NAMES[1], rejected=len(invalid), null_class_included=True)

    g = torch.linspace(-.3, .4, 64, dtype=torch.float64)
    translated = {}
    for kind, model in heads.items():
        other = ClassCenteredUtilityHead(kind).double(); other.load_state_dict(initial)
        with torch.no_grad(): other.decoder.add_(g)
        d0 = model(h, roles, mask, details=True); d1 = other(h, roles, mask, details=True)
        translated[kind] = (d0, d1)
        if kind in ('centered_native', 'centered_full'):
            close(d0['logits'], d1['logits']); close(d0['alpha'], d1['alpha'])
        elif kind == 'construction_only':
            close(d0['alpha'], d1['alpha'])
            assert float((d1['logits']-d0['logits']).abs().max()) > 1e-8
    d = source(h, roles, mask, details=True)
    predicted = (d['alpha'] * torch.einsum('btd,d->bt', d['values'], g)[:, None]).sum(-1)
    close(translated['native'][1]['logits']-translated['native'][0]['logits'], predicted)
    record(CHECK_NAMES[2], centered_invariance=True,
           construction_only_logit_change=float((translated['construction_only'][1]['logits']-translated['construction_only'][0]['logits']).abs().max()))

    for kind in ('centered_native', 'centered_full', 'construction_only'):
        model = heads[kind]; model.zero_grad(set_to_none=True)
        z, _ = model(h, roles, mask); F.cross_entropy(z, torch.tensor([7])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if kind != 'construction_only':
            assert float(model.decoder.grad.sum(0).abs().max()) <= 2e-9
    record(CHECK_NAMES[3], decoder_null_dimensions=64)

    psi, phi = states((1, 3, 8)), states((40, 8))
    u = (torch.randn((1, 40, 3), dtype=torch.float64)*.3).requires_grad_()
    direction = torch.zeros_like(u); direction[0, 0, 0] = 1.; direction[0, 11, 1] = -.4
    weights = torch.linspace(-.4, .7, 40, dtype=torch.float64)[None]
    def objective(value):
        return (class_centered_measurement(psi, phi, value, mask)['aggregate']*weights).sum()
    grad = torch.autograd.grad(objective(u), u)[0]
    step = 1e-6
    fd = (objective(u+step*direction)-objective(u-step*direction))/(2*step)
    close((grad*direction).sum(), fd, 2e-6)
    w, _ = class_centered_utilities(u, mask)
    detached_loss = (measured_attention(psi, phi, w, mask,
        construction_utilities=w.detach())['aggregate']*weights).sum()
    detached_grad = torch.autograd.grad(detached_loss, u)[0]
    assert float((grad-detached_grad).abs().max()) > 1e-8
    close(grad.sum(1), torch.zeros((1, 3), dtype=torch.float64), 2e-9)
    record(CHECK_NAMES[4], finite_difference_error=float(((grad*direction).sum()-fd).abs()),
           detached_construction_gradient_difference=float((grad-detached_grad).abs().max()))

    # Normalized-state witness, not an original-circuit or task-data fixture.
    ps = torch.zeros((1, 2, 8), dtype=torch.complex128); ps[0, 0, 0] = 1; ps[0, 1, 1] = 1
    ph = torch.zeros((40, 8), dtype=torch.complex128)
    ph[:, 0] = .5; ph[:, 1] = 3**.5/2
    ph[0, 0] = 3**.5/2; ph[0, 1] = .5
    mm = torch.ones((1, 2), dtype=torch.bool)
    u0 = torch.zeros((1, 40, 2), dtype=torch.float64); u0[:, :, 1] = 4.
    u0[0, 0, 0] = 1.; u0[0, 1, 0] = -1.
    prob = torch.einsum('rd,btd->brt', ph.conj(), ps).abs().square()
    alpha = (prob+EPS)/(prob+EPS).sum(-1, keepdim=True)
    cw, _ = class_centered_utilities(u0, mm)
    bias = torch.full((40,), -100., dtype=torch.float64); bias[:2] = 0
    zn = (alpha*u0).sum(-1)+bias; zc = (alpha*cw).sum(-1)+bias
    close((zc[:, 0]-zc[:, 1]), torch.ones(1, dtype=torch.float64))
    close((zn[:, 0]-zn[:, 1]), torch.full((1,), 1-2/(1+2*EPS), dtype=torch.float64))
    assert F.cross_entropy(zc, torch.tensor([0])) < F.cross_entropy(zn, torch.tensor([0]))
    assert F.cross_entropy(zc, torch.tensor([1])) > F.cross_entropy(zn, torch.tensor([1]))
    record(CHECK_NAMES[5], scope='Abstract normalized-state native routing; all40 classes with fixed other-class biases.')

    for tokens in (1, 3):
        sp, sq = states((1, tokens, 8)), states((40, 8))
        m = torch.ones((1, tokens), dtype=torch.bool)
        uv = torch.linspace(-.2, .3, tokens, dtype=torch.float64)[None, None].expand(1, 40, tokens).clone().requires_grad_()
        z = class_centered_measurement(sp, sq, uv, m)
        close(z['centered_utilities'], torch.zeros_like(uv))
        close(z['alpha'], measured_attention(sp, sq, torch.zeros_like(uv), m, eta=0)['alpha'])
        loss = (z['aggregate']*weights).sum()
        ug = torch.autograd.grad(loss, uv)[0]
        assert torch.isfinite(ug).all()
    record(CHECK_NAMES[6])

    for kind in KINDS:
        first = heads[kind](h, roles, mask)
        reload = ClassCenteredUtilityHead(kind).double(); reload.load_state_dict(heads[kind].state_dict())
        second = reload(h, roles, mask)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
    try: ClassCenteredUtilityHead('centered_simplex')
    except ValueError: pass
    else: raise AssertionError('Undeclared kind accepted')
    record(CHECK_NAMES[7], same_state_replay=True, new_checkpoint_files=0)
    assert len(checks) == len(CHECK_NAMES)
    return dict(status='CLASS_CENTERED_NUMERICAL_CANARY_COMPLETE', checks=checks,
        check_count=len(checks), seed=20261119, task_training=False, optimizer_steps=0,
        real_dataset_reads=0, new_checkpoints=0, COS_requests=0, cloud_writes=0,
        independent_confirmation=False, formal_handoff=False, L5='NOT_ACCEPTED',
        qualification='Synthetic float64 CPU implementation checks only; no utility, novelty or CaseStudy-readiness claim.')
