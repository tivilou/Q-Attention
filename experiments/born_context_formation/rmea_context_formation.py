"""One contextualization block; frozen task inputs and forty-class output.

No scheduler, data loader, classifier bypass, or training policy lives here.
Final dephasing removes both terminal state coherences, not circuit gates.
"""
import math
import torch
from torch import nn
from baselines.retacred.rmea import circuit

MODES = ('born_context', 'dephased_context', 'dot_context',
         'rbf_context', 'pair_mlp_context')
EPS = 1e-6
COMMON = ('value.weight', 'value.bias')  # Same-shaped native initialization only.
SHARED = COMMON + ('context.weight', 'context.bias',
                   'classifier.weight', 'classifier.bias')


class ContextFormationHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown contextualization condition')
        self.mode = mode
        width = 12 if mode in MODES[:2] else 64
        self.query = nn.Linear(1028, width)
        # Dot key bias cancels under row softmax; omit that unused direction.
        self.key = nn.Linear(1028, width, bias=mode != 'dot_context')
        if mode == 'pair_mlp_context':
            # A scalar output bias also cancels under row softmax.
            self.pair = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1, bias=False))
        self.value = nn.Linear(1024, 64)
        self.context = nn.Linear(64, 64)
        self.classifier = nn.Linear(128, 40)

    def forward(self, h, roles, mask, details=False, final_dephase=False):
        if h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,):
            raise ValueError('Expected h[B,T,1024] and roles[B,T,4]')
        if mask.dtype != torch.bool or mask.shape != h.shape[:2] or not mask.any(-1).all():
            raise ValueError('Each example needs valid words')
        if final_dephase and self.mode not in MODES[:2]:
            raise ValueError('Terminal dephasing requires quantum preparation')
        logits, rows = [], []
        for hi, ri, mi in zip(h, roles, mask):
            # Exclude padding before every projection and gate, including NaNs.
            hv, rv = hi[mi], ri[mi]
            subject, obj = rv[:, 2] > 0, rv[:, 3] > 0
            if not subject.any() or not obj.any():
                raise ValueError('Valid words must include both entity memberships')
            x = torch.cat((hv, rv), -1)
            q, k = self.query(x), self.key(x)
            qs = ks = None
            if self.mode in MODES[:2]:
                qs = circuit((math.pi * torch.tanh(q)).reshape(-1, 2, 3, 2))
                ks = circuit((math.pi * torch.tanh(k)).reshape(-1, 2, 3, 2))
                if self.mode == 'dephased_context' or final_dephase:
                    score = qs.abs().square() @ ks.abs().square().T
                else:
                    score = (qs.conj() @ ks.T).abs().square()
                raw = score + EPS
                alpha = raw / raw.sum(-1, keepdim=True)
            else:
                if self.mode == 'dot_context':
                    score = q @ k.T / math.sqrt(64)
                elif self.mode == 'rbf_context':
                    score = -.5 * (q[:, None, :] - k[None, :, :]).square().mean(-1)
                else:
                    n = len(q)
                    pair = torch.cat((q[:, None, :].expand(n, n, 64),
                                      k[None, :, :].expand(n, n, 64)), -1)
                    score = self.pair(pair).squeeze(-1)
                alpha = score.softmax(-1)
            values = self.value(hv)
            hidden = torch.tanh(self.context(alpha @ values))
            readout = torch.cat((hidden[subject].mean(0), hidden[obj].mean(0)))
            logits.append(self.classifier(readout))
            if details:
                rows.append(dict(score=score, alpha=alpha, hidden=hidden,
                                 q_state=qs, k_state=ks))
        output = torch.stack(logits)
        return dict(logits=output, rows=rows) if details else output


def make_models(seed=13, initial=None):
    # A different condition must not consume another condition's RNG stream.
    result = {}
    with torch.random.fork_rng(devices=[]):
        for mode in MODES:
            torch.manual_seed(seed)
            result[mode] = ContextFormationHead(mode)
    common = result[MODES[0]].state_dict()
    if initial is not None:
        for name in COMMON:
            if common[name].shape != initial[name].shape:
                raise ValueError('Native value initialization shape mismatch')
            common[name] = initial[name].detach().clone()
    for model in result.values():
        state = model.state_dict()
        for name in SHARED:
            state[name] = common[name].detach().clone()
        model.load_state_dict(state, strict=True)
    return result


def real_parameters(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)
