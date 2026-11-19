"""Donor/recipient separation; provisional float64 CPU implementation.

The two folds alternate in valid-token order. Both conditions retain native
fold masses and every token value; self_fold changes only the donor mapping.
No conditional independence or task benefit is inferred by this module.
"""
import math

import torch

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
from experiments.rmea_utility_measurement.head import measured_attention

KINDS = ('self_fold', 'cross_fold')


def fold_attention(psi, phi, utilities, mask, *, kind='cross_fold', eta=1.):
    if kind not in KINDS or not math.isfinite(eta) or eta < 0:
        raise ValueError('Unknown fold condition or invalid eta')
    if (psi.ndim != 3 or psi.shape[-1] != 8 or phi.shape != (40, 8)
            or psi.dtype != torch.complex128 or phi.dtype != psi.dtype
            or utilities.dtype != torch.float64 or psi.device.type != 'cpu'):
        raise ValueError('Expected complex128 CPU states and all40 float64 utilities')
    b, t, _ = psi.shape
    if (b < 1 or t < 1 or utilities.shape != (b, 40, t)
            or mask.shape != (b, t) or mask.dtype != torch.bool
            or not (psi.device == phi.device == utilities.device == mask.device)
            or not mask.any(-1).all()):
        raise ValueError('Nonempty matching batch/token axes and boolean mask required')
    if (not torch.isfinite(psi[mask]).all() or not torch.isfinite(phi).all()
            or not torch.isfinite(utilities.masked_select(mask[:, None])).all()):
        raise ValueError('Nonfinite valid state or utility')
    for norms in (psi.abs().square().sum(-1)[mask], phi.abs().square().sum(-1)):
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-10, rtol=0):
            raise ValueError('Unnormalized state; no silent repair')
    states = torch.where(mask[..., None], psi, torch.zeros_like(psi))
    values = torch.where(mask[:, None], utilities, torch.zeros_like(utilities))
    raw = torch.einsum('rd,btd->brt', phi.conj(), states).abs().square() + EPS
    raw = torch.where(mask[:, None], raw, torch.zeros_like(raw))
    native = raw / raw.sum(-1, keepdim=True)
    fold_b = mask & ((mask.to(torch.int64).cumsum(-1) - 1).remainder(2) == 1)
    fold_a = mask & ~fold_b
    rows, diagnostics = [], []
    for i in range(b):
        masks = (fold_a[i], fold_b[i])
        masses = torch.stack([native[i, :, m].sum(-1) for m in masks])
        count = int(mask[i].sum())
        # Only the one-token case has an empty fold. For two/three tokens,
        # evaluate both effects: cross_fold is mathematically native, whereas
        # self_fold can still redistribute a two-token fold at length three.
        if count == 1:
            rows.append(native[i])
            diagnostics.append(dict(short_native=True, native_fold_masses=masses,
                                    donor_updates=None))
            continue
        updates = []
        for m in masks:
            sm = states[i:i+1, m]
            updates.append(measured_attention(sm, phi, values[i:i+1, :, m],
                torch.ones(sm.shape[:2], dtype=torch.bool, device=sm.device), eta=eta))
        row = native[i] * 0
        for recipient, m in enumerate(masks):
            donor = recipient if kind == 'self_fold' else 1 - recipient
            sp = states[i, m]
            rho = sp[..., :, None] * sp.conj()[..., None, :]
            weight = torch.einsum('rij,tji->rt', updates[donor]['effect'][0], rho).real
            alpha = masses[recipient, :, None] * weight / weight.sum(-1, keepdim=True)
            row = row.index_copy(-1, torch.where(m)[0], alpha)
        rows.append(row)
        diagnostics.append(dict(short_native=False, native_fold_masses=masses,
                                donor_updates=updates))
    alpha = torch.stack(rows)
    aggregate = (alpha * values).sum(-1)
    if not torch.isfinite(alpha).all() or not torch.isfinite(aggregate).all():
        raise FloatingPointError('Nonfinite fold result; no clipping or retry')
    return dict(alpha=alpha, aggregate=aggregate, native_alpha=native,
                fold_masks=(fold_a, fold_b), samples=diagnostics)


class CrossFoldUtilityHead(RMEAHead):
    """Original stored parameters; ordinary gradients through native fold masses."""
    def __init__(self, kind='cross_fold', *, eta=1.):
        if kind not in KINDS or not math.isfinite(eta) or eta < 0:
            raise ValueError('Unknown fold condition or invalid eta')
        super().__init__('quantum')
        self.kind, self.update_eta = kind, eta

    def forward(self, h, roles, mask, details=False):
        if h.dtype != torch.float64 or h.device.type != 'cpu':
            raise ValueError('This provisional adapter supports float64 CPU only')
        original = super().forward(h, roles, mask, details=True)
        update = fold_attention(original['psi'], original['phi'],
            original['utilities'], mask, kind=self.kind, eta=self.update_eta)
        logits = update['aggregate'] + self.bias
        if not details:
            return logits, update['alpha']
        return dict(logits=logits, alpha=update['alpha'],
                    raw_utilities=original['utilities'], psi=original['psi'],
                    phi=original['phi'], update=update)
