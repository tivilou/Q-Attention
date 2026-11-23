"""Shared-null differential readout; no scheduler, storage or training policy."""
import math
import torch
from torch import nn
from baselines.retacred.rmea import RMEAHead
from baselines.retacred.frozen_feature_head import FrozenFeatureHead

MODES = ('coherent', 'diagonal', 'classical', 'native')


def contrast(alpha, utilities, bias):
    """Null-reference margins. Two learned effects, not success/failure of one."""
    weights = alpha - alpha[:, :1]
    values = utilities - utilities[:, :1]
    return (weights * values).sum(-1) + bias - bias[:1]


class SharedNullHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown scientific condition')
        self.mode = mode
        self.head = (FrozenFeatureHead('relation_softmax') if mode == 'classical'
                     else RMEAHead('dephased' if mode == 'diagonal' else 'quantum'))

    def forward(self, h, roles, mask, details=False):
        if self.mode == 'classical':
            a = self.head
            key = a.key(torch.cat((h, roles), -1))
            score = torch.einsum('btd,rd->brt', key, a.query) / math.sqrt(8)
            alpha = score.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            values = a.value(h)
            utilities = torch.einsum('btd,rd->brt', values, a.decoder)
            native = (alpha * utilities).sum(-1) + a.bias
        else:
            d = self.head(h, roles, mask, details=True)
            alpha, utilities, native = d['alpha'], d['utilities'], d['logits']
        logits = native if self.mode == 'native' else contrast(alpha, utilities, self.head.bias)
        if details:
            return dict(logits=logits, alpha=alpha, utilities=utilities,
                        signed_weights=alpha-alpha[:, :1])
        return logits


def initialize(initial, seed):
    """Build once before lanes; all arms share value/decoder/bias and batch order."""
    heads = {}
    for mode in MODES:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = SharedNullHead(mode).double()
        if mode == 'classical':
            state = model.head.state_dict()
            for key in ('value.weight', 'value.bias', 'decoder', 'bias'):
                state[key] = initial[key].clone()
            model.head.load_state_dict(state)
        else:
            model.head.load_state_dict(initial)
        heads[mode] = model
    return heads


def fixtures(models, inputs):
    """Only the changed readout boundary; the actual first fit checks gradients."""
    errors = {}

    def check(name, value):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError('Nonfinite fixture residual: ' + name)
        errors[name] = x

    for mode, cpu_model in models.items():
        model = SharedNullHead(mode).double().to(inputs[0].device)
        model.load_state_dict(cpu_model.state_dict())
        d = model(*inputs, details=True)
        if mode == 'native':
            direct = model.head(*inputs)[0]
        else:
            direct = torch.stack([((d['alpha'][:, r]-d['alpha'][:, 0]) *
                (d['utilities'][:, r]-d['utilities'][:, 0])).sum(-1)
                + model.head.bias[r]-model.head.bias[0] for r in range(40)], -1)
            check(mode+'_zero_mass', d['signed_weights'].sum(-1).abs().max())
            check(mode+'_null_zero', d['logits'][:, 0].abs().max())
            shifted = contrast(d['alpha'], d['utilities']+7.25, model.head.bias)
            check(mode+'_constant_value_shift', (shifted-d['logits']).abs().max())
        check(mode+'_formula', (direct-d['logits']).abs().max())
        g1 = torch.autograd.grad(d['logits'].square().mean(), tuple(model.parameters()), retain_graph=True)
        g2 = torch.autograd.grad(direct.square().mean(), tuple(model.parameters()))
        for i, (a, b) in enumerate(zip(g1, g2)):
            check(mode+'_gradient_'+str(i), (a-b).abs().max())
        from torch.nn.functional import pad
        padded = (pad(inputs[0], (0,0,0,2), value=.123),
                  pad(inputs[1], (0,0,0,2), value=1.), pad(inputs[2], (0,2), value=False))
        check(mode+'_padding', (model(*padded)-d['logits']).abs().max())
    if max(errors.values()) >= 1e-8:
        raise ValueError('Readout fixture failed')
    return dict(status='PASS', errors=errors,
                scope='Formula/autograd/padding/constant-shift; no resume/COS matrix')
