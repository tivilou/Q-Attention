"""Equivalent direct linear utilities; no new attention operation or features."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import EPS, circuit
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum', 'dephased', 'gaussian_base')
SHARED = ('value.weight', 'value.bias', 'decoder', 'bias')


class FoldedHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        super().__init__('real12' if kind == 'gaussian_base' else kind)
        self.kind = kind
        # Retain baseline initialization tensors as nontraining buffers solely
        # for the accepted core's initial-state identity and source parity.
        for name in SHARED:
            parent, field = (self.head.value, name.split('.')[1]) if name.startswith('value.') else (self.head, name)
            value = parent._parameters.pop(field).detach().clone()
            parent.register_buffer(field, value)
        self.utility = nn.Linear(1024, 40)

    def fold_initial_utilities(self):
        a = self.head
        with torch.no_grad():
            self.utility.weight.copy_(a.decoder @ a.value.weight)
            self.utility.bias.copy_(a.bias + a.decoder @ a.value.bias)

    def attention(self, h, roles, mask):
        a = self.head
        if self.kind == 'gaussian_base':
            key = a.key(torch.cat((h, roles), -1))
            scale = (key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6) / 24).sqrt()
            p = torch.exp(-((key[:, None] - a.query[None, :, None]) * scale).square().sum(-1))
        else:
            angles = math.pi * torch.tanh(a.prepare(torch.cat((h, roles), -1))).reshape(h.shape[:2] + (2, 3, 2))
            psi = circuit(angles)
            phi = circuit(math.pi * torch.tanh(a.effect))
            p = (torch.einsum('rd,btd->brt', phi.conj(), psi).abs().square()
                 if self.kind == 'quantum' else
                 torch.einsum('rd,btd->brt', phi.abs().square(), psi.abs().square()))
        raw = (p + EPS) * mask[:, None]
        return raw / raw.sum(-1, keepdim=True)

    def forward(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Bad native frozen-word inputs')
        alpha = self.attention(h, roles, mask)
        utilities = F.linear(h, self.utility.weight)
        return (alpha * utilities.transpose(1, 2)).sum(-1) + self.utility.bias


def make_head(kind):
    return FoldedHead(kind)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        m = make_head(kind)
        native = initialize_head('real12' if kind == 'gaussian_base' else kind,
                                 initial, dict(seed=seed, training_dtype='float32'))
        m.head.load_state_dict(native.head.state_dict())
        m.fold_initial_utilities()
        models[kind] = m
    return models
