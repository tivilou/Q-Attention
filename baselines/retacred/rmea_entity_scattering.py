"""Exploratory ordered-terminal readout; q is an upstream diagnostic, not pooling attention.

Fixed v1 constants are prospective numerical choices, not fitted hyperparameters.
Both arms retain the original RMEA parameters and original linear Born routing.
"""
import torch
from .rmea import RMEAHead
from .rmea_token_transport import route

LAMBDA = 1.
ENERGY = 0.
ETA = .1
GAMMA = 1.
READOUT_EPS = 1e-12
KINDS = ('complex', 'real_magnitude')


def anchors(roles, mask):
    """First valid word in each supplied entity span; no label or text heuristic."""
    if roles.shape != mask.shape+(4,) or mask.dtype != torch.bool:
        raise ValueError('Expected four roles and a valid-word mask')
    if not torch.isfinite(roles).all():
        raise ValueError('Nonfinite role features')
    flags = roles[..., 2:]
    if not ((flags == 0) | (flags == 1)).all():
        raise ValueError('Entity flags must be binary')
    spans = flags.bool() & mask[..., None]
    if not spans.any(1).all() or (spans[..., 0] & spans[..., 1]).any():
        raise ValueError('Need nonempty disjoint entity spans')
    return spans[..., 0].long().argmax(-1), spans[..., 1].long().argmax(-1)


def scattering(psi, q, onsite, mask, subject, obj, *, kind='complex', dense=False):
    """H=lambda sqrt(q)(G-I)sqrt(q)+diag(onsite), with equal terminal leads.

    `dense` is a numerical reference for complex hopping. Real-magnitude
    hopping generally has no eight-dimensional factorization and uses a dense
    solve. Padding is decoupled. The ordered amplitude is R[subject, object].
    """
    if kind not in KINDS or psi.ndim != 3 or psi.shape[-1] != 8:
        raise ValueError('Bad scattering kind or state shape')
    b, t, _ = psi.shape
    if q.ndim != 3 or q.shape[0] != b or q.shape[2] != t or q.shape[1] < 1 or onsite.shape != q.shape:
        raise ValueError('Expected q and onsite [B,R,T]')
    if (psi.dtype != torch.complex128 or q.dtype != torch.float64 or onsite.dtype != torch.float64
            or mask.dtype != torch.bool or mask.shape != (b, t)
            or any(x.device != psi.device for x in (q, onsite, mask, subject, obj))):
        raise ValueError('Expected colocated complex128/float64 inputs')
    if not mask.any(-1).all() or not torch.isfinite(psi[mask]).all() or not torch.isfinite(onsite).all() or not torch.isfinite(q).all():
        raise ValueError('Nonfinite or empty input')
    if not torch.allclose(psi[mask].norm(dim=-1), torch.ones_like(psi[mask].real[..., 0]), atol=1e-10, rtol=0):
        raise ValueError('Nonunit prepared states')
    valid = mask[:, None, :].expand_as(q)
    if (q[valid] <= 1e-300).any() or (q[~valid] != 0).any() or not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=2e-10, rtol=0):
        raise ValueError('q must be strictly positive on valid words and normalized')
    for a in (subject, obj):
        if a.dtype != torch.long or a.shape != (b,) or ((a < 0) | (a >= t)).any() or not mask.gather(1, a[:, None]).all():
            raise ValueError('Invalid readout anchor')
    if (subject == obj).any():
        raise ValueError('Distinct terminals required')
    s = psi.masked_fill(~mask[..., None], 0)
    e = onsite.masked_fill(~valid, 0)
    root = q.clamp_min(1e-300).sqrt().masked_fill(~valid, 0)
    lead = torch.zeros((b, t), dtype=torch.float64, device=psi.device)
    lead.scatter_(1, subject[:, None], GAMMA/2)
    lead.scatter_(1, obj[:, None], GAMMA/2)
    f = ENERGY-e+LAMBDA*q+1j*(ETA+lead[:, None, :])
    # X has columns sqrt(lambda*q_i)*psi_i, so X^H X=lambda D G D.
    x = LAMBDA**.5*s.transpose(-2, -1)[:, None, :, :]*root[..., None, :]
    ix = torch.arange(b, device=psi.device)[:, None]
    ir = torch.arange(q.shape[1], device=psi.device)[None, :]
    if kind == 'real_magnitude' or dense:
        gram = s.conj() @ s.transpose(-2, -1)
        if kind == 'real_magnitude':
            gram = gram.abs().to(psi.dtype)
        a = torch.diag_embed(f)-LAMBDA*root[..., :, None]*gram[:, None, :, :]*root[..., None, :]
        rhs = torch.zeros_like(f)
        rhs.scatter_(-1, obj[:, None, None].expand(b, q.shape[1], 1), 1.)
        col = torch.linalg.solve(a, rhs[..., None]).squeeze(-1)
        amplitude = col.gather(-1, subject[:, None, None].expand(b, q.shape[1], 1)).squeeze(-1)
    else:
        inv = f.reciprocal()
        small = torch.eye(8, dtype=psi.dtype, device=psi.device)-(x*inv[..., None, :]) @ x.mH
        xs = x[ix, ir, :, subject[:, None]]
        xo = x[ix, ir, :, obj[:, None]]
        z = torch.linalg.solve(small, xo[..., None]).squeeze(-1)
        amplitude = inv[ix, ir, subject[:, None]]*(xs.conj()*z).sum(-1)*inv[ix, ir, obj[:, None]]
    transmission = GAMMA**2*amplitude.abs().square()
    logits = (transmission+READOUT_EPS).log()
    if not torch.isfinite(logits).all() or (transmission > 1+1e-9).any():
        raise ValueError('Nonfinite or nonpassive singleton transmission')
    return dict(logits=logits, transmission=transmission, amplitude=amplitude)


class EntityScatteringHead(RMEAHead):
    """Same parameter/state-dict schema, different scoring law; no residual mix."""
    def __init__(self, kind='complex'):
        if kind not in KINDS:
            raise ValueError('Unknown entity scattering kind')
        super().__init__('quantum')
        self.kind = kind

    def forward(self, h, roles, mask, details=False):
        if not torch.isfinite(h).all() or not torch.isfinite(roles).all():
            raise ValueError('Finite features including padding required')
        d = super().forward(h, roles, mask, details=True)
        q = route(d['psi'], d['phi'], mask, kind='linear_coherent')['alpha']
        s, o = anchors(roles, mask)
        e = d['utilities'].tanh()
        result = scattering(d['psi'], q, e, mask, s, o, kind=self.kind)
        z = result['logits']+self.bias
        if not details:
            return z
        return dict(logits=z, q=q, onsite=e, subject=s, obj=o, psi=d['psi'],
                    phi=d['phi'], utilities=d['utilities'], readout=result)
