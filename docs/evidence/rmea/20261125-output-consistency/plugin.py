"""Native Q/D/G with known output-VAT; architecture and evaluation unchanged."""
import torch
from torch import nn
from baselines.retacred.rmea_matched_routers import initialize_head
from gaussian_reference import CorroborationHead
from consistency import backward as consistency_backward

MODES = ('quantum_vat', 'dephased_vat', 'gaussian_vat', 'quantum_random_consistency')


class ConsistencyHead(nn.Module):
    def __init__(self, kind, native, seed=13):
        super().__init__()
        if kind not in MODES:
            raise ValueError(kind)
        self.kind, self.native, self.seed = kind, native, seed
        self.direction_generator = None
        self.objective_totals = {'updates': 0, 'ce': 0., 'kl': 0.,
                                 'initial_zero_norm': 0, 'power_zero_norm': 0}

    @property
    def head(self):
        return self.native.head

    def forward(self, h, roles, mask):
        return self.native(h, roles, mask)


def make_head(kind):
    from baselines.retacred.rmea_matched_routers import Head
    native = (CorroborationHead('gaussian_base') if kind == 'gaussian_vat'
              else Head('dephased' if kind == 'dephased_vat' else 'quantum'))
    return ConsistencyHead(kind, native)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        mode = ('real12' if kind == 'gaussian_vat' else
                'dephased' if kind == 'dephased_vat' else 'quantum')
        native = initialize_head(mode, initial, dict(seed=seed, training_dtype='float32'))
        if kind == 'gaussian_vat':
            native = CorroborationHead('gaussian_base', native)
        models[kind] = ConsistencyHead(kind, native, seed)
    return models


def backward(model, inputs, labels):
    if model.direction_generator is None:
        model.direction_generator = torch.Generator(device=inputs[0].device).manual_seed(model.seed)
    loss, record = consistency_backward(model, inputs, labels,
        generator=model.direction_generator,
        directed=model.kind != 'quantum_random_consistency', alpha=1., epsilon=.01, xi=.001)
    model.objective_totals['updates'] += 1
    for name in ('ce', 'kl', 'initial_zero_norm', 'power_zero_norm'):
        model.objective_totals[name] += float(record[name] or 0)
    return loss
