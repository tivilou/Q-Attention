"""One-sided cue corroboration; unchanged affine value/decoder and native keys."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit, EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_corroboration', 'dephased_corroboration',
         'normalized_dephased_corroboration', 'gaussian_corroboration',
         'gaussian_base', 'real12', 'softmax12')


def scaled_key(key):
    weights = key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6)
    return key * (weights / 24).sqrt()


def sides(roles, mask):
    subject = mask & (roles[..., 0].abs() <= roles[..., 1].abs())
    return subject, mask & ~subject


class CorroborationHead(nn.Module):
    def __init__(self, kind, native=None):
        super().__init__()
        if kind not in MODES:
            raise ValueError(kind)
        self.kind = kind
        self.native = native if native is not None else Head('softmax12' if kind == 'softmax12' else 'real12')

    @property
    def head(self):
        return self.native.head

    def components(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool
                or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word inputs')
        a = self.head
        key = a.key(torch.cat((h, roles), -1))
        k, q = scaled_key(key), scaled_key(a.query)
        base = torch.exp(-(k[:, None]-q[None, :, None]).square().sum(-1))
        subject, obj = sides(roles, mask)
        count = obj.sum(-1).clamp_min(1).to(h.dtype)
        if self.kind == 'gaussian_base':
            gate = torch.ones_like(mask, dtype=h.dtype)
        elif self.kind == 'gaussian_corroboration':
            distance = (k.square().sum(-1)[:, :, None] + k.square().sum(-1)[:, None, :]
                        - 2*torch.einsum('btd,bsd->bts', k, k)).clamp_min(0)
            gate = (torch.exp(-distance)*obj[:, None]).sum(-1)/count[:, None]
        else:
            psi = circuit((math.pi*torch.tanh(key)).reshape(h.shape[:2]+(2, 3, 2)))
            if self.kind == 'quantum_corroboration':
                rho = torch.einsum('bti,btj,bt->bij', psi, psi.conj(), obj.to(psi.dtype))/count[:, None, None]
                gate = torch.einsum('bti,bij,btj->bt', psi.conj(), rho, psi).real
            else:
                population = psi.abs().square()
                if self.kind == 'normalized_dephased_corroboration':
                    population = F.normalize(population, dim=-1)
                mean = (population*obj[..., None]).sum(1)/count[:, None]
                gate = (population*mean[:, None]).sum(-1)
            gate = gate.clamp_min(0)
        # No reference side: return the base router, including singleton inputs.
        gate = torch.where(subject & obj.any(-1)[:, None], gate, torch.ones_like(gate))
        raw = (base*gate[:, None]+EPS)*mask[:, None]
        alpha = raw/raw.sum(-1, keepdim=True)
        return alpha, gate, base

    def forward(self, h, roles, mask):
        if self.kind in ('real12', 'softmax12'):
            return self.native(h, roles, mask)
        alpha, _, _ = self.components(h, roles, mask)
        a = self.head
        return (torch.einsum('brt,btd->brd', alpha, a.value(h))*a.decoder).sum(-1)+a.bias


def make_head(kind):
    return CorroborationHead(kind)


def initialize(initial, seed):
    result = {}
    for kind in MODES:
        native_kind = 'softmax12' if kind == 'softmax12' else 'real12'
        native = initialize_head(native_kind, initial, dict(seed=seed, training_dtype='float32'))
        result[kind] = CorroborationHead(kind, native)
    return result
