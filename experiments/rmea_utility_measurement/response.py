"""R97 local-response checks and frozen R96 model analysis; no fitting."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead, circuit
from experiments.rmea_ecqp_screen import metrics, runner
from experiments.rmea_utility_measurement.head import measured_attention
from experiments.rmea_utility_measurement.train import TrainHead


def checked_close(left, right, tolerance, key, errors):
    assert torch.isfinite(left).all() and torch.isfinite(right).all(), key
    difference = float((left - right).abs().max())
    errors[key] = max(errors.get(key, 0.), difference)
    assert torch.allclose(left, right, atol=tolerance['absolute'], rtol=tolerance['relative']), (key, difference)


def decomposition(psi, phi, u, mask, zero):
    """Matched construction/readout only. Exact normalized-state formula."""
    p = phi[..., :, None] * phi.conj()[..., None, :]
    q = torch.eye(8, dtype=phi.dtype) - p
    clean = torch.where(mask[..., None], psi, torch.zeros_like(psi))
    rho = clean[..., :, None] * clean.conj()[..., None, :]
    s = rho.sum(1)
    denominator = torch.einsum('rij,bji->br', zero['initial_effect'], s).real
    assert (denominator > 0).all()
    a = zero['covariance']
    pa, qa, cross = p[None] @ a @ p[None], q[None] @ a @ q[None], p[None] @ a @ q[None]
    scale = zero['scale'] * denominator
    eplus, eminus = (1 + EPS)/(1 + 8*EPS), EPS/(1 + 8*EPS)
    ell = 1/((1 + 8*EPS)*math.log((1 + EPS)/EPS))
    diagonal = (eplus*pa.abs().square().sum((-1, -2)) + eminus*qa.abs().square().sum((-1, -2)))/scale
    offdiagonal = 2*ell*cross.abs().square().sum((-1, -2))/scale
    return dict(diagonal=diagonal, cross=offdiagonal, full=diagonal + offdiagonal,
        projector_residual=float((p @ p - p).abs().max()),
        log_residual=float((torch.matrix_exp(zero['initial_log']) - zero['initial_effect']).abs().max()))


def response_checks(psi, phi, u, mask, bias, labels, design, errors):
    """All eta evaluations use the unchanged nonnegative-eta source interface."""
    assert not any(v.requires_grad for v in [psi, phi, u, bias])
    zero = measured_attention(psi, phi, u, mask, eta=0.)
    terms = decomposition(psi, phi, u, mask, zero)
    baseline = zero['aggregate'] + bias
    probability = baseline.softmax(-1)
    selected = torch.arange(len(labels))
    h = {}
    for kind, g in [('full', terms['full']), ('pinched', terms['diagonal']), ('cross', terms['cross'])]:
        h[kind] = g[selected, labels] - (probability * g).sum(-1)
    checked_close(h['full'] - h['pinched'], h['cross'], design['tolerances']['identity'], 'CE_cross_gap_identity', errors)
    step = design['derivative_step']
    for kind, key in [('full', 'full'), ('pinched', 'diagonal')]:
        eta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        out = measured_attention(psi, phi, u, mask, eta=eta, kind=kind)
        expected = terms[key]
        # Eta is a scalar shared by all classes/examples, so AD checks SUMs.
        # Per-entry checks below independently use one-sided finite differences.
        with torch.enable_grad():
            derivative_sum = torch.autograd.grad(out['aggregate'].sum(), eta, retain_graph=True)[0]
            derivative_ce = torch.autograd.grad(F.cross_entropy(out['aggregate'] + bias, labels, reduction='sum'), eta)[0]
        checked_close(derivative_sum.detach(), expected.sum(), design['tolerances']['autograd'], kind + '_aggregate_AD_sum', errors)
        checked_close(derivative_ce.detach(), -h[kind].sum(), design['tolerances']['autograd'], kind + '_CE_AD_sum', errors)
        with torch.no_grad():
            first = measured_attention(psi, phi, u, mask, eta=step, kind=kind)['aggregate']
            second = measured_attention(psi, phi, u, mask, eta=2*step, kind=kind)['aggregate']
            fd = (-3*zero['aggregate'] + 4*first - second)/(2*step)
            checked_close(fd, expected, design['tolerances']['finite_difference'], kind + '_aggregate_FD_entries', errors)
            ce0 = F.cross_entropy(baseline, labels, reduction='none')
            ce1 = F.cross_entropy(first + bias, labels, reduction='none')
            ce2 = F.cross_entropy(second + bias, labels, reduction='none')
            checked_close((-3*ce0 + 4*ce1 - ce2)/(2*step), -h[kind], design['tolerances']['finite_difference'], kind + '_CE_FD_entries', errors)
    assert terms['projector_residual'] <= design['tolerances']['geometry_absolute']
    assert terms['log_residual'] <= design['tolerances']['geometry_absolute']
    norm_error = max(float((psi[mask].abs().square().sum(-1) - 1).abs().max()), float((phi.abs().square().sum(-1) - 1).abs().max()))
    return dict(baseline_logits=baseline, terms=terms, h=h, norm_error=norm_error)


def canary(design):
    errors = {}
    # Same source-circuit harmful-CE witness as R94; two fixed labels expose
    # opposite signed competition using the same mathematical construction.
    theta = torch.zeros((1, 2, 2, 3, 2), dtype=torch.float64)
    theta[0, 0, 0, 0, 0], theta[0, 1, 0, 0, 0] = math.pi/2, -math.pi/2
    psi = circuit(theta)
    phi = circuit(torch.zeros((40, 2, 3, 2), dtype=torch.float64))
    u = torch.zeros((1, 40, 2), dtype=torch.float64)
    u[0, 1], u[0, 2] = torch.tensor([.1, -.1], dtype=torch.float64), torch.tensor([1., -1.], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    bias = torch.full((40,), -10., dtype=torch.float64)
    bias[1:3] = 0
    harmful = response_checks(psi, phi, u, mask, bias, torch.tensor([1]), design, errors)
    helpful = response_checks(psi, phi, u, mask, bias, torch.tensor([2]), design, errors)
    assert float(harmful['h']['full'][0]) < -1e-5 and float(helpful['h']['full'][0]) > 1e-5
    assert float(harmful['terms']['cross'].max()) > 1e-5
    # Exact orthogonality lies outside the F>0 asymptotic proposition.
    orthogonal = torch.zeros((1, 2, 8), dtype=torch.complex128)
    orthogonal[0, 0, 1], orthogonal[0, 1, 2] = 1, 1
    query = torch.zeros((40, 8), dtype=torch.complex128)
    query[:, 0] = 1
    utilities = torch.tensor([1., -1.], dtype=torch.float64).expand(1, 40, 2)
    degenerate = response_checks(orthogonal, query, utilities, mask, bias, torch.tensor([1]), design, errors)
    checked_close(degenerate['terms']['diagonal'], torch.full((1, 40), 1/math.sqrt(3), dtype=torch.float64), design['tolerances']['identity'], 'F_zero_counterexample', errors)
    checked_close(degenerate['terms']['cross'], torch.zeros((1, 40), dtype=torch.float64), design['tolerances']['identity'], 'F_zero_cross', errors)
    constant = response_checks(orthogonal, query, torch.full((1, 40, 2), .25, dtype=torch.float64), mask, bias, torch.tensor([1]), design, errors)
    checked_close(constant['terms']['full'], torch.zeros((1, 40), dtype=torch.float64), design['tolerances']['identity'], 'constant_utility_response', errors)
    return dict(status='PASS', checks=errors, harmful_CE_prime=-float(harmful['h']['full'][0]),
        helpful_CE_prime=-float(helpful['h']['full'][0]), F_zero_counterexample=True,
        constant_utility=True, positive_eta_interface=True, negative_eta_calls=0)


def new_summary():
    return dict(count=0, sum=0., minimum=None, maximum=None, positive=0, negative=0, near_zero=0)


def accumulate(row, tensor, cutoff):
    v = tensor.detach()
    assert torch.isfinite(v).all()
    row['count'] += v.numel()
    row['sum'] += float(v.sum())
    minimum, maximum = float(v.min()), float(v.max())
    row['minimum'] = minimum if row['minimum'] is None else min(row['minimum'], minimum)
    row['maximum'] = maximum if row['maximum'] is None else max(row['maximum'], maximum)
    row['positive'] += int((v > cutoff).sum())
    row['negative'] += int((v < -cutoff).sum())
    row['near_zero'] += int((v.abs() <= cutoff).sum())


def run(a, design):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(design['seed'])
    assert EPS == 1e-6
    fixtures = canary(design)
    identity = design['data']
    data = runner.read_training_prefix(a.assets, a.receipt, a.cohort, identity['cohort_sha256'], a.donors,
        identity['donors_sha256'], expected_receipt_sha256=identity['prefix_receipt_sha256'], expected_plan_sha256=identity['plan_sha256'])
    ids = data['splits']['eval']
    assert len(ids) == 972 and len({data['cohort'][int(i)]['group'] for i in ids}) == 866
    models = {}
    for kind in design['conditions']:
        model = TrainHead(kind).double()
        state = torch.load(a.checkpoints/(kind + '-final.pt'), map_location='cpu', weights_only=True)
        assert all(v.dtype == torch.float64 for v in state.values())
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
        before = runner.state_fingerprint(model.state_dict())
        assert sum(p.numel() for p in model.parameters()) == 81028
        summaries = {n: new_summary() for n in ['G_full', 'G_diagonal', 'G_cross', 'H_full', 'H_pinched', 'H_cross']}
        errors = {}
        maximum_norm = maximum_projector = maximum_log = zero_error = 0.
        predictions = {n: [] for n in ['eta0', 'deployed']}
        losses = {n: 0. for n in predictions}
        for start in range(0, len(ids), design['batch_size']):
            h, roles, mask, y, donor = runner.batch(data, ids[start:start + design['batch_size']], 'eval', False)
            assert donor is None
            h, roles = h.double(), roles.double()
            with torch.no_grad():
                d = RMEAHead.forward(model, h, roles, mask, details=True)
            observed = response_checks(d['psi'], d['phi'], d['utilities'], mask, model.bias, y, design, errors)
            zero = observed['baseline_logits']
            checked_close(zero, d['logits'], design['tolerances']['identity'], 'native_eta0_logits', errors)
            zero_error = max(zero_error, float((zero - d['logits']).abs().max()))
            maximum_norm = max(maximum_norm, observed['norm_error'])
            maximum_projector = max(maximum_projector, observed['terms']['projector_residual'])
            maximum_log = max(maximum_log, observed['terms']['log_residual'])
            for name in ['full', 'diagonal', 'cross']:
                accumulate(summaries['G_' + name], observed['terms'][name], design['sign_cutoff'])
            for name in ['full', 'pinched', 'cross']:
                accumulate(summaries['H_' + name], observed['h'][name], design['sign_cutoff'])
            with torch.no_grad():
                deployed = model(h, roles, mask)[0]
                for name, z in [('eta0', zero), ('deployed', deployed)]:
                    assert torch.isfinite(z).all()
                    predictions[name].extend(z.argmax(-1).tolist())
                    losses[name] += float(F.cross_entropy(z, y, reduction='sum'))
        assert runner.state_fingerprint(model.state_dict()) == before
        gold = np.asarray(data['targets'][ids], dtype=np.int64)
        measured = {name: metrics.from_confusion(metrics.confusion(gold, np.asarray(pred, dtype=np.int64))) | dict(ce=losses[name]/len(ids)) for name, pred in predictions.items()}
        expected = design['reported_metrics'][kind]
        assert measured['deployed']['macro_f1'] == expected['macro_f1']
        assert measured['deployed']['accuracy'] == expected['accuracy']
        assert abs(measured['deployed']['ce'] - expected['ce']) <= design['tolerances']['reported_ce_absolute']
        for row in summaries.values():
            row['mean'] = row.pop('sum')/row['count']
            assert row['positive'] + row['negative'] + row['near_zero'] == row['count']
        models[kind] = dict(metrics=measured, summaries=summaries, checks=errors,
            checkpoint_unchanged=True, state_fingerprint=before, maximum_norm_error=maximum_norm,
            maximum_projector_residual=maximum_projector, maximum_log_residual=maximum_log,
            maximum_native_eta0_logit_error=zero_error,
            deployed_minus_eta0=dict(macro_f1=measured['deployed']['macro_f1'] - measured['eta0']['macro_f1'],
                ce=measured['deployed']['ce'] - measured['eta0']['ce']),
            eta0_scope='Hypothetical common utility measurement response at this model weights; simplex deployed readout uses its different tilt.')
    return dict(status='UTILITY_LOCAL_RESPONSE_DIAGNOSTIC_COMPLETE', decision='ALGEBRA_AND_REPLAY_CHECKS_PASS_DESCRIPTIVE_ONLY',
        canary=fixtures, models=models, conditions=design['conditions'], evaluation_examples=972, groups=866,
        derivative_step=design['derivative_step'], derivative_path='eta0 matched construction/readout; scalar AD sums plus per-entry one-sided FD',
        task_training=False, optimizer_steps=0, new_checkpoints=0, COS_requests=0, cloud_writes=0,
        independent_confirmation=False, formal_handoff=False, L5='NOT_ACCEPTED',
        scope='Reused original-training development group partition, three R96 final weights; no training-causality or finite-eta theorem claim.',
        elapsed_seconds=time.monotonic() - started)
