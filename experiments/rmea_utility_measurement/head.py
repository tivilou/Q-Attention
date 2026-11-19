"""R93 v002 utility-to-measurement path; diagnostic float64 implementation.

No optimizer, hidden detaches, classical residual, eigensystem gradients or
second epsilon. The original RMEA source and checkpoints remain unchanged.
"""
import hashlib
import math

import torch

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead


def utility_permutation(ids, mask):
    """Input-ID-only, shared class mapping; padding stays fixed."""
    if mask.ndim != 2 or mask.dtype != torch.bool or len(ids) != len(mask):
        raise ValueError('Explicit IDs and a boolean token mask are required')
    result = []
    for sample, valid in zip(ids, mask):
        if not isinstance(sample, str) or not sample:
            raise ValueError('Nonempty input identity required')
        positions = torch.where(valid)[0].tolist()
        if not positions:
            raise ValueError('Empty sample')
        sources = sorted(positions, key=lambda j: hashlib.sha256(
            f'r93-utility-v2:{sample}:{j}'.encode()).digest())
        if len(sources) > 1 and sources == positions:
            sources = sources[1:] + sources[:1]
        row = list(range(len(valid)))
        for dest, source in zip(positions, sources):
            row[dest] = source
        result.append(row)
    return torch.tensor(result, dtype=torch.long, device=mask.device)


def measured_attention(psi, phi, utilities, mask, *, eta=1., kind='full',
                       construction_utilities=None):
    """Return tensors with full autograd lineage; no learned parameter is added.

    psi[B,T,8], phi[R,8], utilities[B,R,T], mask[B,T]. Current numerical
    qualification is deliberately float64/complex128 CPU-only.
    """
    if kind not in ('full', 'pinched') or not math.isfinite(eta) or eta < 0:
        raise ValueError('Unknown update or invalid eta')
    if psi.ndim != 3 or psi.shape[-1] != 8 or phi.ndim != 2 or phi.shape[-1] != 8:
        raise ValueError('Three-qubit states required')
    b, t, d = psi.shape
    if b < 1 or t < 1 or len(phi) < 1:
        raise ValueError('Empty batch, token or class axis')
    if utilities.shape != (b, len(phi), t) or mask.shape != (b, t):
        raise ValueError('Token/class axes differ')
    if psi.dtype != torch.complex128 or phi.dtype != psi.dtype or utilities.dtype != torch.float64:
        raise ValueError('This canary implementation is float64/complex128 only')
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Nonempty boolean mask required')
    if not (psi.device == phi.device == utilities.device == mask.device):
        raise ValueError('Device mismatch')
    cu = utilities if construction_utilities is None else construction_utilities
    if cu.shape != utilities.shape or cu.dtype != utilities.dtype or cu.device != utilities.device:
        raise ValueError('Construction utility mismatch')
    if not torch.isfinite(psi[mask]).all() or not torch.isfinite(phi).all():
        raise ValueError('Nonfinite valid state')
    if not torch.isfinite(utilities.masked_select(mask[:, None, :])).all() or not torch.isfinite(cu.masked_select(mask[:, None, :])).all():
        raise ValueError('Nonfinite valid utility')
    for norms in [(psi.abs().square().sum(-1))[mask], phi.abs().square().sum(-1)]:
        if not torch.allclose(norms, torch.ones_like(norms), rtol=0, atol=2e-10):
            raise ValueError('States are not normalized; no silent repair')
    psi = torch.where(mask[..., None], psi, torch.zeros_like(psi))
    u = torch.where(mask[:, None, :], utilities, torch.zeros_like(utilities))
    cu = torch.where(mask[:, None, :], cu, torch.zeros_like(cu))
    rho = psi[..., :, None] * psi.conj()[..., None, :]
    p = phi[..., :, None] * phi.conj()[..., None, :]
    eye = torch.eye(d, dtype=psi.dtype, device=psi.device)
    e0 = (p + EPS * eye) / (1 + d * EPS)
    log0 = math.log((1 + EPS)/(1 + d * EPS))*p + math.log(EPS/(1 + d * EPS))*(eye-p)
    s = rho.sum(1)
    operator = torch.einsum('brt,btij->brij', cu.to(psi.dtype), rho)
    denominator = torch.einsum('rij,bji->br', e0, s).real
    f0 = torch.einsum('rij,brji->br', e0, operator).real / denominator
    a = operator - f0[..., None, None] * s[:, None]
    # tr(A^2)=||A||_F^2 on Hermitian matrices. This form avoids an imaginary
    # roundoff component without clipping or modifying A.
    scale = torch.sqrt(1 + a.abs().square().sum((-1, -2)))
    update = a / scale[..., None, None]
    if kind == 'pinched':
        q = eye-p
        update = p[None] @ update @ p[None] + q[None] @ update @ q[None]
    exponential = torch.matrix_exp(log0[None] + eta * update)
    effect = exponential / exponential.diagonal(dim1=-2, dim2=-1).sum(-1).real[..., None, None]
    raw = torch.einsum('brij,btji->brt', effect, rho).real
    alpha = raw / raw.sum(-1, keepdim=True)
    aggregate = (alpha*u).sum(-1)
    if not torch.isfinite(effect).all() or not torch.isfinite(alpha).all() or not torch.isfinite(aggregate).all():
        raise FloatingPointError('Nonfinite update; no clipping or retry')
    return dict(alpha=alpha, aggregate=aggregate, effect=effect, initial_effect=e0,
                initial_log=log0, covariance=a, scale=scale, construction_aggregate=f0)


class UtilityMeasurementHead(RMEAHead):
    """Same trainable state as the original; no original-mode dispatch change."""
    def __init__(self, *, eta=1., kind='full'):
        super().__init__('quantum')
        self.update_eta = eta
        self.update_kind = kind

    def forward(self, h, roles, mask, details=False, *, permutation=None):
        original = super().forward(h, roles, mask, details=True)
        utilities = original['utilities']
        construction = None if permutation is None else utilities.gather(-1, permutation[:, None].expand_as(utilities))
        update = measured_attention(original['psi'], original['phi'], utilities, mask,
            eta=self.update_eta, kind=self.update_kind, construction_utilities=construction)
        logits = update['aggregate'] + self.bias
        if not details:
            return logits, update['alpha']
        return dict(logits=logits, alpha=update['alpha'], utilities=utilities,
            construction_utilities=utilities if construction is None else construction,
            psi=original['psi'], phi=original['phi'], update=update)
