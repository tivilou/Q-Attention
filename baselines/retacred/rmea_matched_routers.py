"""Original endpoints and width12 controls; no optimizer, storage or scheduler code.

Width12 formulas come from the reviewed 20261124 matched-router screen.
Registered parameter counts do not establish equal effective capacity.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import RMEAHead

MODES = ('uniform', 'quantum', 'dephased', 'softmax12', 'real12')
SHARED = ('value.weight', 'value.bias', 'decoder', 'bias')


def parameter_schema(mode):
    if mode not in MODES:
        raise ValueError(mode)
    shapes = {'head.value.weight': [64, 1024], 'head.value.bias': [64],
              'head.decoder': [40, 64], 'head.bias': [40]}
    if mode in ('quantum', 'dephased'):
        shapes.update({'head.prepare.weight': [12, 1028], 'head.prepare.bias': [12],
                       'head.effect': [40, 2, 3, 2]})
    elif mode != 'uniform':
        shapes.update({'head.key.weight': [12, 1028], 'head.key.bias': [12],
                       'head.query': [40, 12]})
    return shapes


class Head(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.head = RMEAHead(mode if mode in MODES[:3] else 'uniform')
        if mode in ('softmax12', 'real12'):
            self.head.key = nn.Linear(1028, 12)
            self.head.query = nn.Parameter(torch.randn(40, 12) / math.sqrt(12))

    def forward(self, h, roles, mask):
        a = self.head
        if self.mode in MODES[:3]:
            return a(h, roles, mask)[0]
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool
                or not mask.any(-1).all()):
            raise ValueError('Invalid shared frozen-word input')
        key = a.key(torch.cat((h, roles), -1))
        if self.mode == 'softmax12':
            score = torch.einsum('btd,rd->brt', key, a.query) / math.sqrt(12)
            alpha = score.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        else:
            score = torch.einsum('btd,rd->brt', F.normalize(key, dim=-1),
                                 F.normalize(a.query, dim=-1)).square()
            raw = (score + 1e-6) * mask[:, None]
            alpha = raw / raw.sum(-1, keepdim=True)
        return (torch.einsum('brt,btd->brd', alpha, a.value(h)) * a.decoder).sum(-1) + a.bias


def initialize_head(mode, initial, cfg):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'])
        model = Head(mode).to(dtype=getattr(torch, cfg.get('training_dtype', 'float64')))
    if mode in ('quantum', 'dephased'):
        model.head.load_state_dict(initial)
    else:
        state = model.head.state_dict()
        state.update({k: initial[k].clone() for k in SHARED})
        model.head.load_state_dict(state)
    return model


def validate_initialized_head(model, mode, initial, cfg):
    state = model.state_dict()
    shapes = parameter_schema(mode)
    if cfg['parameter_schemas'][mode] != shapes or set(state) != set(shapes):
        raise ValueError('Declared model parameter inventory differs')
    dtype = getattr(torch, cfg.get('training_dtype', 'float64'))
    if any(list(t.shape) != shapes[k] or t.dtype != dtype for k, t in state.items()):
        raise ValueError('Model shape or dtype differs')
    expected = 68200 if mode == 'uniform' else 81028
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError('Registered parameter count differs')
    if any(not torch.equal(model.head.state_dict()[k], initial[k]) for k in SHARED):
        raise ValueError('Shared initial values differ')
    if mode in ('quantum', 'dephased'):
        if any(not torch.equal(v, initial[k]) for k, v in model.head.state_dict().items()):
            raise ValueError('Original quantum endpoint initialization differs')
