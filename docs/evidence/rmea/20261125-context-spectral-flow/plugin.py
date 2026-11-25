"""Mean-state unitary flow; no training, scheduler, storage or Git code."""
import math
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit, EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_flow', 'real12_flow', 'quantum', 'dephased')


class FlowHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        super().__init__('real12' if kind == 'real12_flow' else 'quantum' if kind == 'quantum_flow' else kind)
        self.kind = kind
        self.time = math.pi

    def states(self, h, roles):
        a = self.head
        if self.kind == 'quantum_flow':
            key = circuit((math.pi * torch.tanh(a.prepare(torch.cat((h, roles), -1)))).reshape(h.shape[:2] + (2, 3, 2)))
            query = circuit(math.pi * torch.tanh(a.effect))
        else:
            key = F.normalize(a.key(torch.cat((h, roles), -1)), dim=-1)
            query = F.normalize(a.query, dim=-1)
            dtype = torch.complex128 if h.dtype == torch.float64 else torch.complex64
            key, query = key.to(dtype), query.to(dtype)
        return key, query

    def attention(self, h, roles, mask, exponential=torch.matrix_exp):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word inputs')
        key, query = self.states(h, roles)
        count = mask.sum(-1).to(h.dtype)
        mean = torch.einsum('bti,btj,bt->bij', key, key.conj(), mask.to(key.dtype)) / count[:, None, None]
        unitary = exponential(-1j * self.time * mean)
        moved = torch.einsum('bij,rj->bri', unitary, query)
        p = torch.einsum('bri,bti->brt', moved.conj(), key).abs().square()
        native = torch.einsum('ri,bti->brt', query.conj(), key).abs().square()
        raw = (p + EPS) * mask[:, None]
        baseline = (native + EPS) * mask[:, None]
        return raw / raw.sum(-1, keepdim=True), raw.sum(-1), baseline.sum(-1)

    def forward(self, h, roles, mask):
        if self.kind in ('quantum', 'dephased'):
            return super().forward(h, roles, mask)
        alpha, _, _ = self.attention(h, roles, mask)
        a = self.head
        return (torch.einsum('brt,btd->brd', alpha, a.value(h)) * a.decoder).sum(-1) + a.bias


def make_head(kind):
    return FlowHead(kind)


def initialize(initial, seed):
    result = {}
    for kind in MODES:
        model = FlowHead(kind)
        native_kind = 'real12' if kind == 'real12_flow' else 'quantum' if kind == 'quantum_flow' else kind
        native = initialize_head(native_kind, initial, dict(seed=seed, training_dtype='float32'))
        model.load_state_dict(native.state_dict())
        result[kind] = model
    return result
