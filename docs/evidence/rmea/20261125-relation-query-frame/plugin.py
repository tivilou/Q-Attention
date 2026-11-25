"""Relation-frame effects on unfiltered keys; no lifecycle or data access."""
import math
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit, EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_relation_frame', 'dephased_relation_frame', 'real12_relation_frame')
NATIVE = dict(zip(MODES, ('quantum', 'dephased', 'real12')))


class InvSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        eigen, v = torch.linalg.eigh((x+x.mH)/2)
        if not torch.isfinite(eigen).all() or not (eigen > 0).all():
            raise ValueError('Nonpositive or nonfinite relation frame')
        s = eigen.sqrt()
        ctx.save_for_backward(s, v)
        return (v/s.unsqueeze(-2)) @ v.mH

    @staticmethod
    def backward(ctx, gradient):
        s, v = ctx.saved_tensors
        g = (gradient+gradient.mH)/2
        divided = -1/(s[:, None]*s[None, :]*(s[:, None]+s[None, :]))
        out = v @ ((v.mH @ g @ v)*divided) @ v.mH
        return (out+out.mH)/2


def frame_probability(key, query, diagonal=False):
    """FP64 internal frame algebra; return original real precision, no extra EPS."""
    output_dtype = key.real.dtype
    if diagonal:
        dtype = torch.complex128 if key.is_complex() else torch.float64
        p, q = key.to(dtype).abs().square(), query.to(dtype).abs().square()
        denominator = q.sum(0)+query.shape[0]*EPS
        raw = torch.einsum('rd,btd->brt', (q+EPS)/denominator, p)
    else:
        dtype = torch.complex128 if key.is_complex() else torch.float64
        k, q = key.to(dtype), query.to(dtype)
        identity = torch.eye(q.shape[-1], device=q.device, dtype=dtype)
        gram = q.T @ q.conj()+q.shape[0]*EPS*identity
        inverse = InvSqrt.apply(gram)
        transformed_key = torch.einsum('ij,btj->bti', inverse, k)
        raw = torch.einsum('rd,btd->brt', q.conj(), transformed_key).abs().square()
        raw = raw+EPS*transformed_key.abs().square().sum(-1)[:, None, :]
    return raw.to(output_dtype)


class RelationFrameHead(Head):
    def __init__(self, kind):
        if kind not in NATIVE:
            raise ValueError(kind)
        super().__init__(NATIVE[kind])
        self.kind = kind

    def forward(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool
                or not mask.any(-1).all()):
            raise ValueError('Invalid nonempty frozen-word inputs')
        a = self.head
        if self.mode == 'real12':
            key = F.normalize(a.key(torch.cat((h, roles), -1)), dim=-1)
            query = F.normalize(a.query, dim=-1)
        else:
            angles = math.pi*torch.tanh(a.prepare(torch.cat((h, roles), -1)))
            key = circuit(angles.reshape(h.shape[:2]+(2, 3, 2)))
            query = circuit(math.pi*torch.tanh(a.effect))
        probability = frame_probability(key, query, self.mode == 'dephased')
        raw = probability*mask[:, None]
        alpha = raw/raw.sum(-1, keepdim=True)
        pooled = torch.einsum('brt,btd->brd', alpha, a.value(h))
        return (pooled*a.decoder).sum(-1)+a.bias


def make_head(kind):
    return RelationFrameHead(kind)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        m = make_head(kind)
        native = initialize_head(NATIVE[kind], initial, dict(seed=seed, training_dtype='float32'))
        m.load_state_dict(native.state_dict(), strict=True)
        models[kind] = m
    return models
