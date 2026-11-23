"""Trainable-count-matched controls; effective degrees of freedom differ."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import RMEAHead

MODES = ('quantum', 'dephased', 'softmax12', 'real12')
SHARED = ('value.weight', 'value.bias', 'decoder', 'bias')


class Head(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        if mode in ('quantum', 'dephased'):
            self.head = RMEAHead(mode)
        else:
            self.head = RMEAHead('uniform')
            self.head.key = nn.Linear(1028, 12)
            self.head.query = nn.Parameter(torch.randn(40, 12) / math.sqrt(12))

    def forward(self, h, roles, mask):
        if self.mode in ('quantum', 'dephased'):
            return self.head(h, roles, mask)[0]
        a = self.head
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


def make_head(mode):
    return Head(mode)


def initialize(initial, seed):
    models = {}
    for mode in MODES:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = Head(mode).double()
        state = model.head.state_dict()
        if mode in ('quantum', 'dephased'):
            model.head.load_state_dict(initial)
        else:
            for key in SHARED:
                state[key] = initial[key].clone()
            model.head.load_state_dict(state)
        assert sum(p.numel() for p in model.parameters()) == 81028
        models[mode] = model
    assert torch.equal(models['softmax12'].head.key.weight, models['real12'].head.key.weight)
    assert torch.equal(models['softmax12'].head.query, models['real12'].head.query)
    return models


def fixtures(models, inputs):
    errors = {}
    for mode, original in models.items():
        model = Head(mode).double().to(inputs[0].device)
        model.load_state_dict(original.state_dict())
        h, roles, mask = inputs
        got = model(h, roles, mask)
        a = model.head
        if mode in ('quantum', 'dephased'):
            expected = a(h, roles, mask)[0]
        else:
            k = a.key(torch.cat((h, roles), -1))
            if mode == 'softmax12':
                s = (k[:, None] * a.query[None, :, None]).sum(-1) / math.sqrt(12)
                w = s.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            else:
                s = (F.normalize(k, dim=-1)[:, None] * F.normalize(a.query, dim=-1)[None, :, None]).sum(-1).square()
                w = (s + 1e-6) * mask[:, None]
                w = w / w.sum(-1, keepdim=True)
            u = (a.value(h)[:, None] * a.decoder[None, :, None]).sum(-1)
            expected = (w * u).sum(-1) + a.bias
        errors[mode + '_formula'] = float((got - expected).abs().max())
        x = torch.autograd.grad(got.square().mean(), tuple(model.parameters()), retain_graph=True)
        z = torch.autograd.grad(expected.square().mean(), tuple(model.parameters()))
        errors[mode + '_gradient'] = max(float((u-v).abs().max()) for u, v in zip(x, z))
        grads = dict(zip(dict(model.named_parameters()), x))
        if mode == 'softmax12':
            errors['softmax12_inert_key_bias'] = float(grads['head.key.bias'].abs().max())
        elif mode == 'dephased':
            ix = torch.tensor([7, 9, 11], device=h.device)
            errors['dephased_inert_last_RZ_prepare_weight'] = float(grads['head.prepare.weight'][ix].abs().max())
            errors['dephased_inert_last_RZ_prepare_bias'] = float(grads['head.prepare.bias'][ix].abs().max())
            errors['dephased_inert_last_RZ_effect'] = float(grads['head.effect'][:, 1, :, 1].abs().max())
        padded = (F.pad(h, (0, 0, 0, 2), value=.123), F.pad(roles, (0, 0, 0, 2), value=1.), F.pad(mask, (0, 2), value=False))
        errors[mode + '_padding'] = float((model(*padded) - got).abs().max())
    assert all(math.isfinite(v) and v < 1e-8 for v in errors.values())
    return dict(status='PASS', errors=errors, scope='Count/formula/gradients/padding and known invariant directions only; no lifecycle/COS matrix.',
                parameter_limit='All81028 registered trainable scalars, not equal effective capacity. Softmax12 has12 invariant keybias scalars; dephased has3207 last-RZ invariant scalars. Other continuous reparameterization symmetries not exhausted.')


def summarize(rows, logits, labels, cfg):
    import numpy as np
    assert all(x['parameters'] == 81028 for x in rows.values())
    q = rows['quantum']
    contrasts = {k: dict(macro_delta=q['positive39_macro_f1']-v['positive39_macro_f1'],
                         ce_comparator_minus_quantum=v['ce']-q['ce'])
                 for k, v in rows.items() if k != 'quantum'}
    gates = {k: v['macro_delta'] > 0 for k, v in contrasts.items()}
    return dict(results=rows, contrasts=contrasts, gates=gates,
        gate='PASS_FOR_FORMAL_STUDY_DESIGN_ONLY' if all(gates.values()) else 'NO_L3_PROMOTION_FROM_THIS_SCREEN',
        interpretation='One exposed TRAIN split and seed, fixed1024 updates. This measures operational ordering, not population significance, mature utility, novelty or impossibility. Both positive and negative results retained; no budget/seed/split/control rescue.',
        L3='NOT_ACCEPTED', L4='NOT_ACCEPTED', L5='NOT_ACCEPTED')
