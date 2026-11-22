"""Protected-context binary effects, revision 1; fixed-rank first derivative.

Constrained minimum-variance filtering is known. This module implements a
prospective task hypothesis, with no utility, originality or L5 certificate.
"""
import math

import torch
from torch.autograd.function import once_differentiable

from .rmea import EPS, RMEAHead, circuit

RIDGE = 1 / 8
CUTOFF = 1e-10
KINDS = ('native', 'protected_context', 'context_scrambled', 'unprotected_context')


class _SpanProjector(torch.autograd.Function):
    """Differentiate the projector, avoiding derivatives of individual vectors.

    On a fixed-rank stratum with a positive retained/discarded spectral gap,
    dP = U [F * (U^H d(ZZ^H) U)] U^H, where cross-cluster entries are
    F_ij = (keep_i-keep_j)/(s_i^2-s_j^2) and within-cluster entries are zero.
    Repeated singular values within either cluster are allowed. Rank crossings,
    a closing cross-cluster gap, and second derivatives are outside this rule.
    """
    @staticmethod
    def forward(ctx, z):
        if z.shape[-2] != 8 or z.shape[-1] < 8 or not z.is_complex():
            raise ValueError('Expected complex [...,8,M>=8] protected columns')
        left, singular, _ = torch.linalg.svd(z, full_matrices=False)
        keep = singular > CUTOFF * singular[..., :1]
        projector = (left * keep.unsqueeze(-2)) @ left.mH
        ctx.save_for_backward(z, left, singular.square(), keep)
        ctx.mark_non_differentiable(singular)
        return projector, singular

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_projector, _grad_singular):
        z, left, eigenvalues, keep = ctx.saved_tensors
        incoming = (grad_projector + grad_projector.mH) / 2
        cross = keep.unsqueeze(-1) != keep.unsqueeze(-2)
        gap = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)
        # Mask BEFORE division: repeated values in one cluster must not form 0/0.
        denominator = torch.where(cross, gap, torch.ones_like(gap))
        numerator = keep.to(gap.dtype).unsqueeze(-1) - keep.to(gap.dtype).unsqueeze(-2)
        divided_difference = torch.where(cross, numerator / denominator,
                                         torch.zeros_like(gap))
        rotated = left.mH @ incoming @ left
        hermitian = left @ (divided_difference * rotated) @ left.mH
        return 2 * hermitian @ z


def protected_projector(psi, mention):
    """Return the retained entity-span projector and detached singular values."""
    z = (psi * mention.unsqueeze(-1)).transpose(-2, -1)
    if z.shape[-1] < 8:
        z = torch.nn.functional.pad(z, (0, 8-z.shape[-1]))
    return _SpanProjector.apply(z)


class ProtectedContextHead(RMEAHead):
    """Same trainable state as RMEAHead('quantum'); no donor/batch coupling.

    The fixed DFT covariance scramble is a prospective control, distinct from
    the earlier frozen-probe donor condition. It preserves the covariance
    eigenvalues and trace, while changing its basis relative to this example.
    """
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError('Unknown protected-context condition')
        super().__init__('quantum')
        self.kind = kind
        coordinate = torch.arange(8, dtype=torch.float64)
        dft = torch.exp(-2j*math.pi*coordinate[:, None]*coordinate[None, :]/8)/math.sqrt(8)
        self.register_buffer('_dft', dft, persistent=False)

    def forward(self, h, roles, mask, details=False):
        if self.kind == 'native':
            return super().forward(h, roles, mask, details=details)
        if h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,):
            raise ValueError('Bad features')
        if mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError('Bad valid mask')
        mention = mask & ((roles[..., 2] != 0) | (roles[..., 3] != 0))
        if not mention.any(-1).all():
            raise ValueError('Each example needs a protected entity mention')
        context = mask & ~mention
        angles = math.pi*torch.tanh(self.prepare(torch.cat((h, roles), -1)))
        angles = angles.reshape(h.shape[:2]+(2, 3, 2))
        effect_angles = math.pi*torch.tanh(self.effect)
        # The frozen probe's 1e-10 rank cutoff requires complex128 SVD/solve.
        # Keep this cast differentiable; return attention in the source dtype.
        psi = circuit(angles).to(torch.complex128)
        phi = circuit(effect_angles).to(torch.complex128)
        count = context.sum(-1).clamp_min(1)
        covariance = psi.transpose(-2, -1) @ (psi*context.unsqueeze(-1)).conj()
        covariance = covariance/count[:, None, None]
        used_covariance = covariance
        if self.kind == 'context_scrambled':
            dft = self._dft.to(dtype=psi.dtype)
            used_covariance = dft @ covariance @ dft.mH
        projector, singular = protected_projector(psi, mention)
        eye = torch.eye(8, device=psi.device, dtype=psi.dtype)
        perp = eye-projector if self.kind != 'unprotected_context' else eye.expand_as(projector)
        system = perp @ used_covariance @ perp + RIDGE*eye
        rhs = -perp @ used_covariance @ phi.T
        correction = torch.linalg.solve(system, rhs).transpose(-2, -1)
        corrected = phi.unsqueeze(0)+correction
        amplitude = corrected.conj() @ psi.transpose(-2, -1)
        probability = amplitude.abs().square()
        raw = (probability+EPS)*mask[:, None, :]
        normalizer = raw.sum(-1, keepdim=True)
        alpha = (raw/normalizer).to(h.dtype)
        values = self.value(h)
        utilities = torch.einsum('btd,rd->brt', values, self.decoder)
        logits = (alpha*utilities).sum(-1)+self.bias
        if not details:
            return logits, alpha
        return dict(logits=logits, alpha=alpha, values=values, utilities=utilities,
                    angles=angles, effect_angles=effect_angles, psi=psi, phi=phi,
                    mention=mention, covariance=covariance, used_covariance=used_covariance,
                    projector=projector, singular_values=singular, perp=perp,
                    correction=correction, corrected=corrected, raw=raw,
                    probability=probability, normalizer=normalizer)
