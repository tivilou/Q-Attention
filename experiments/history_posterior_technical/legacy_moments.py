"""Connected value moments under one router; no parallel classical classifier."""
import math
import torch
from torch import nn
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_covariance', 'gaussian_covariance', 'quantum_mean_product',
         'gaussian_mean_product', 'dephased_covariance')


def native_kind(kind):
    if kind not in MODES:
        raise ValueError(kind)
    return 'real12' if kind.startswith('gaussian') else 'dephased' if kind.startswith('dephased') else 'quantum'


class MomentHead(Head):
    def __init__(self, kind):
        super().__init__(native_kind(kind))
        self.kind = kind
        self.moment_decoder = nn.Parameter(torch.zeros(40, 32))

    def components(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word inputs')
        a = self.head
        if self.kind.startswith('gaussian'):
            key = a.key(torch.cat((h, roles), -1))
            scale = (key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6) / 24).sqrt()
            k, q = key * scale, a.query * scale
            raw = (torch.exp(-(k[:, None] - q[None, :, None]).square().sum(-1)) + EPS) * mask[:, None]
            alpha = raw / raw.sum(-1, keepdim=True)
            values = a.value(h)
            mean = torch.einsum('brt,btd->brd', alpha, values)
            base = (mean * a.decoder).sum(-1) + a.bias
        else:
            detail = a(h, roles, mask, details=True)
            alpha, values, base = detail['alpha'], detail['values'], detail['logits']
            mean = torch.einsum('brt,btd->brd', alpha, values)
        return alpha, values, mean, base

    def statistic(self, alpha, values, mean):
        product_of_means = mean[..., 0::2] * mean[..., 1::2]
        if self.kind.endswith('mean_product'):
            return product_of_means
        mean_of_products = torch.einsum('brt,btj->brj', alpha, values[..., 0::2] * values[..., 1::2])
        return mean_of_products - product_of_means

    def forward(self, h, roles, mask):
        alpha, values, mean, base = self.components(h, roles, mask)
        connected = self.statistic(alpha, values, mean)
        return base + (connected * self.moment_decoder).sum(-1)


def make_head(kind):
    return MomentHead(kind)


def initialize(initial, seed):
    result = {}
    for kind in MODES:
        model = MomentHead(kind)
        native = initialize_head(native_kind(kind), initial, dict(seed=seed, training_dtype='float32'))
        model.head.load_state_dict(native.head.state_dict())
        result[kind] = model
    return result
