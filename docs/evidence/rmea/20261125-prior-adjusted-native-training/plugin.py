"""Known Balanced Softmax training, native factorized Q/D/G readout at inference."""
import math
import torch
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum', 'dephased', 'gaussian_base')
FIT_COUNTS = (5400, 183, 88, 127, 3, 12, 13, 51, 79, 10, 27, 12, 43, 208, 18, 61, 17, 10, 41, 27, 10, 21, 27, 3, 1, 8, 26, 299, 754, 44, 13, 21, 11, 24, 24, 44, 7, 9, 37, 379)


class PriorAdjustedHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        super().__init__('real12' if kind == 'gaussian_base' else kind)
        self.kind = kind
        self.register_buffer('fit_counts', torch.tensor(FIT_COUNTS, dtype=torch.long))

    def raw_logits(self, h, roles, mask):
        if self.kind != 'gaussian_base':
            return super().forward(h, roles, mask)
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid native frozen-word inputs')
        a = self.head
        key = a.key(torch.cat((h, roles), -1))
        scale = (key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6) / 24).sqrt()
        k, q = key * scale, a.query * scale
        p = torch.exp(-(k[:, None]-q[None, :, None]).square().sum(-1))
        raw = (p + EPS) * mask[:, None]
        alpha = raw / raw.sum(-1, keepdim=True)
        return (torch.einsum('brt,btd->brd', alpha, a.value(h)) * a.decoder).sum(-1) + a.bias

    def forward(self, h, roles, mask):
        logits = self.raw_logits(h, roles, mask)
        # Core calls train() for every update and eval() before final assessment.
        # Addition here implements the declared loss without changing the core.
        return logits + self.fit_counts.to(logits).log() if self.training else logits


def make_head(kind):
    return PriorAdjustedHead(kind)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        m = make_head(kind)
        native = initialize_head('real12' if kind == 'gaussian_base' else kind,
                                 initial, dict(seed=seed, training_dtype='float32'))
        m.head.load_state_dict(native.head.state_dict())
        models[kind] = m
    return models
