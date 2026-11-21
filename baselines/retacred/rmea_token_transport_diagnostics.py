"""Fixed-state, label-blind geometric certificates and posthoc utility links.

These diagnostics never change the trained model or the task gate. Per-example
values stay in the private run; only aggregates enter the Git report.
"""
import math
import torch
from torch.nn import functional as F
from .rmea import EPS
from .rmea_token_transport import KINDS, route

TOLERANCE = 1e-8


def gauge_gap(psi, unitary):
    """Sufficient obstruction to UA=AV up to fixed token phases; not utility."""
    a = psi.conj()
    return ((unitary @ a).abs().square().sum(-1) - a.abs().square().sum(-1)).abs().max()


@torch.no_grad()
def inspect_unit(psi, phi, utilities, bias, target):
    """One unpadded fixed example. Target is used only after all route logits."""
    n = len(psi)
    mask = torch.ones(1, n, dtype=torch.bool, device=psi.device)
    routes = {k: route(psi[None], phi, mask, kind=k) for k in KINDS}
    alpha = {k: v['alpha'][0] for k, v in routes.items()}
    logits = {k: (v * utilities).sum(-1) + bias for k, v in alpha.items()}
    eye = torch.eye(8, dtype=torch.complex128, device=psi.device)
    projectors = psi[:, :, None] * psi.conj()[:, None, :]
    b = projectors.sum(0) + n * EPS * eye
    eigenvalues, vectors = torch.linalg.eigh(b)
    if not torch.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
        raise ValueError('Invalid positive Born denominator')
    inverse = (vectors * eigenvalues.rsqrt()[None, :]) @ vectors.mH

    def bounds(effects):
        w = inverse @ effects @ inverse
        result = torch.linalg.eigvalsh((w + w.mH) / 2)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite geometric bound')
        return result

    native_effects = projectors + EPS * eye
    dephased_effects = torch.einsum(
        'tj,jab->tab', routes['cubic_initial_dephased']['unitary'][0].abs().square().to(torch.complex128),
        projectors) + EPS * eye
    ceilings = {'native': bounds(native_effects)[:, -1],
                'cubic_initial_dephased': bounds(dephased_effects)[:, -1]}
    native_readout = bounds(torch.einsum('rt,tab->rab', utilities.to(torch.complex128), native_effects))
    rows = {}
    for kind in KINDS:
        r = routes[kind]
        a = alpha[kind]
        if kind == 'native':
            effects = native_effects
        elif kind == 'cubic_initial_dephased':
            effects = dephased_effects
        else:
            effective = r['unitary'][0].conj() @ psi
            effects = effective[:, :, None] * effective.conj()[:, None, :] + EPS * eye
        denominator = torch.einsum('rd,de,re->r', phi.conj(), b, phi)
        numerator = torch.einsum('rd,tde,re->rt', phi.conj(), effects, phi)
        direct = numerator.real / denominator.real[:, None]
        residual = float((direct - a).abs().max())
        if (denominator.imag.abs().max() > 1e-10 or numerator.imag.abs().max() > 1e-10
                or not math.isfinite(residual) or residual > TOLERANCE):
            raise ValueError('Quadratic attention replay differs')
        z = logits[kind] - bias
        own = bounds(torch.einsum('rt,tab->rab', utilities.to(torch.complex128), effects))
        if max(float((own[:, 0] - z).max()), float((z - own[:, -1]).max())) > TOLERANCE:
            raise ValueError('Actual readout outside its own bound')
        gap = 0. if kind == 'native' else float(gauge_gap(psi, r['unitary'][0]))
        if kind == 'linear_coherent' and gap > TOLERANCE:
            raise ValueError('Linear unitary-absorption null failed')
        rows[kind] = dict(
            native_coordinate_exceedances=int((a - ceilings['native'][None] > TOLERANCE).sum()),
            max_native_coordinate_excess=float((a - ceilings['native'][None]).max()),
            native_readout_upper_exceedances=int((z - native_readout[:, -1] > TOLERANCE).sum()),
            native_readout_lower_exceedances=int((native_readout[:, 0] - z > TOLERANCE).sum()),
            row_norm_gap=gap, phase_gauge_obstruction=gap > TOLERANCE,
            gauge_scope='underlying unitary only; dephased map is not an amplitude map',
            quadratic_score_residual=residual)

    # Targets enter only this post-generation attribution description.
    if type(target) is not int or not 0 <= target < 40:
        raise ValueError('Posthoc target outside fixed labels')
    y = torch.tensor([target], device=psi.device)
    task = {}
    for kind, z in logits.items():
        rivals = z.clone(); rivals[target] = -torch.inf
        task[kind] = dict(ce=float(F.cross_entropy(z[None], y)),
                          true_minus_rival=float(z[target] - rivals.max()))
    excess = (alpha['cubic_coherent'] > ceilings['native'][None] + TOLERANCE) & (
        alpha['cubic_coherent'] > ceilings['cubic_initial_dephased'][None] + TOLERANCE)
    witnesses = []
    for relation, token in excess.nonzero().tolist():
        value = float(utilities[relation, token])
        weights = {k: float(a[relation, token]) for k, a in alpha.items()}
        witnesses.append(dict(relation=relation, token=token, utility=value, weights=weights,
            signed_contribution_change={k: (weights['cubic_coherent'] - v) * value
                                        for k, v in weights.items() if k != 'cubic_coherent'}))
    return dict(tokens=n, coordinates=40*n, routing=rows, simultaneous_ceiling_witnesses=witnesses,
                posthoc_task=task, target=target, tolerance=TOLERANCE,
                claim_limit='Fixed keys/state only; no causal mediation or necessity established')


def aggregate(units):
    """Only counts and maxima; no sample IDs, tokens, targets or predictions."""
    if not units:
        raise ValueError('Missing predeclared diagnostic units')
    return dict(units=len(units), coordinates=sum(u['coordinates'] for u in units),
        readout_unit_relation_denominator=40*len(units),
        simultaneous_ceiling_witnesses=sum(len(u['simultaneous_ceiling_witnesses']) for u in units),
        tolerance=TOLERANCE,
        routing={k: dict(
            native_coordinate_exceedances=sum(u['routing'][k]['native_coordinate_exceedances'] for u in units),
            native_readout_upper_exceedances=sum(u['routing'][k]['native_readout_upper_exceedances'] for u in units),
            native_readout_lower_exceedances=sum(u['routing'][k]['native_readout_lower_exceedances'] for u in units),
            phase_gauge_obstruction_units=sum(u['routing'][k]['phase_gauge_obstruction'] for u in units),
            maximum_row_norm_gap=max(u['routing'][k]['row_norm_gap'] for u in units),
            maximum_quadratic_score_residual=max(u['routing'][k]['quadratic_score_residual'] for u in units))
            for k in KINDS},
        claim_limit='Exploratory geometry only; null inconclusive; no task-gate substitution')
