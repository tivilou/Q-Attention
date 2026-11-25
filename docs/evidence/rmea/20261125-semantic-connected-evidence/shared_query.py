"""Fixed semantic query sharing; native key, circuit and factorized readout."""
from pathlib import Path
import hashlib
import json
import math
import torch
from torch import nn
from baselines.retacred.rmea import EPS, circuit
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_shared', 'dephased_shared', 'gaussian_shared', 'quantum_permuted')
CODE_SHA = 'b7622c7bf8c4076fa695bbe43f0c218b430dedf0f33cb9474a8cddf36c316e6f'
_raw = (Path(__file__).parent / 'label-codes.json').read_bytes()
assert hashlib.sha256(_raw).hexdigest() == CODE_SHA
CODES = json.loads(_raw)


class SharedQueryHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        self.native_mode = ('real12' if kind == 'gaussian_shared' else
                            'dephased' if kind == 'dephased_shared' else 'quantum')
        super().__init__(self.native_mode)
        self.kind = kind
        self.source_query_name = 'query' if self.native_mode == 'real12' else 'effect'
        # Preserve original seed initialization as a buffer; live forward uses code @ W.
        source = self.head._parameters.pop(self.source_query_name)
        self.head.register_buffer(self.source_query_name, source.detach().clone())
        key = 'permuted' if kind == 'quantum_permuted' else 'semantic'
        self.register_buffer('code', torch.tensor(CODES[key], dtype=torch.float32))
        self.query_weights = nn.Parameter(torch.zeros(8, 12))

    def mapped_query(self):
        return self.code @ self.query_weights

    def logits_with_query(self, h, roles, mask, query):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool
                or not mask.any(-1).all() or query.shape != (40, 12)):
            raise ValueError('Invalid frozen-word/query inputs')
        a = self.head
        if self.native_mode == 'real12':
            key = a.key(torch.cat((h, roles), -1))
            scale = (key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6) / 24).sqrt()
            k, q = key * scale, query * scale
            p = torch.exp(-(k[:, None] - q[None, :, None]).square().sum(-1))
        else:
            angles = math.pi * torch.tanh(a.prepare(torch.cat((h, roles), -1)))
            psi = circuit(angles.reshape(h.shape[:2] + (2, 3, 2)))
            phi = circuit(math.pi * torch.tanh(query.reshape(40, 2, 3, 2)))
            if self.native_mode == 'quantum':
                p = torch.einsum('rd,btd->brt', phi.conj(), psi).abs().square()
            else:
                p = torch.einsum('rd,btd->brt', phi.abs().square(), psi.abs().square())
        raw = (p + EPS) * mask[:, None]
        alpha = raw / raw.sum(-1, keepdim=True)
        utilities = torch.einsum('btd,rd->brt', a.value(h), a.decoder)
        return (alpha * utilities).sum(-1) + a.bias

    def forward(self, h, roles, mask):
        return self.logits_with_query(h, roles, mask, self.mapped_query())


def make_head(kind):
    return SharedQueryHead(kind)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        m = make_head(kind)
        native = initialize_head(m.native_mode, initial,
                                 dict(seed=seed, training_dtype='float32'))
        m.head.load_state_dict(native.head.state_dict())
        source = getattr(m.head, m.source_query_name).reshape(40, 12)
        # Use the actual FP32 code, promoted to CPU FP64; never targets or outcomes.
        weights = torch.linalg.lstsq(m.code.double(), source.double(), driver='gelsd').solution
        with torch.no_grad():
            m.query_weights.copy_(weights.to(m.query_weights))
        models[kind] = m
    return models
