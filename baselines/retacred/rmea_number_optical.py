"""Fixed number-coherent optical routing; accepted projection/readout reused."""
import math
import torch
from .rmea_matched_routers import (Head, initialize_head as native_initialize,
                                   validate_initialized_head as native_validate)

MODES = ('number_coherent02', 'number_mixed02',
         'number_mixed02_purity_normalized', 'curvature_matched_gaussian02')


def mode_probability(delta, kind):
    """delta ends in the real/imaginary coordinates of one displacement."""
    x, y = delta.unbind(-1)
    s = x.square() + y.square()
    if kind == 'curvature_matched_gaussian02':
        return torch.exp(-(3 - math.sqrt(2)) * x.square()
                         - (3 + math.sqrt(2)) * y.square())
    # Multiply the exponential before the polynomial's high powers. This
    # avoids a polynomial overflow at ordinary large, finite distances.
    a = torch.exp(-s / 2)
    sa = s * a
    ssa = s * sa
    if kind == 'number_coherent02':
        amplitude = a - sa + ssa / 4 + (x.square() - y.square()) * a / math.sqrt(2)
        return amplitude.square()
    if kind in ('number_mixed02', 'number_mixed02_purity_normalized'):
        d22 = a - 2 * sa + ssa / 2
        probability = (a.square() + d22.square() + sa.square()) / 4
        return probability * (2 if kind.endswith('purity_normalized') else 1)
    raise ValueError(kind)


def optical_probability(key, query, kind):
    k = key.reshape(key.shape[:2] + (6, 2))
    q = query.reshape((40, 6, 2))
    delta = (k[:, None] - q[None, :, None]) / math.sqrt(24)
    return mode_probability(delta, kind).prod(-1)


class NumberOpticalHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        super().__init__('real12')
        self.kind = kind

    def forward(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool
                or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word input')
        a = self.head
        key = a.key(torch.cat((h, roles), -1))
        raw = (optical_probability(key, a.query, self.kind) + 1e-6) * mask[:, None]
        alpha = raw / raw.sum(-1, keepdim=True)
        return (torch.einsum('brt,btd->brd', alpha, a.value(h)) * a.decoder).sum(-1) + a.bias


def initialize_head(kind, initial, cfg):
    native = native_initialize('real12', initial, cfg)
    with torch.random.fork_rng(devices=[]):
        model = NumberOpticalHead(kind).to(dtype=getattr(torch, cfg['training_dtype']))
    model.load_state_dict(native.state_dict(), strict=True)
    return model


def validate_initialized_head(model, kind, initial, cfg):
    mapped = dict(cfg, parameter_schemas={'real12': cfg['parameter_schemas'][kind]})
    native_validate(model, 'real12', initial, mapped)
