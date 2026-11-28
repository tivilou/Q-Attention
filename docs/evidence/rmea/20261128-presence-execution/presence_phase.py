"""Standalone presence/identity readout for the original Re-TACRED40 inputs."""
import math

import torch
from torch import nn


MODES = ('phase_presence_q', 'phase_presence_block_d', 'phase_presence_real16')
COMMON = ('value.weight', 'value.bias', 'decoder', 'bias')


def sector_ids():
    """Map wire 0, before the last CNOT ring, into final computational indices."""
    result = [0] * 8
    for original in range(8):
        output = original
        for control, target in ((0, 1), (1, 2), (2, 0)):
            if output & (1 << (2 - control)):
                output ^= 1 << (2 - target)
        result[output] = original // 4
    return result


class PresenceHead(nn.Module):
    def __init__(self, mode, circuit, eps=1e-6):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unexpected presence mode')
        self.mode = mode
        self.circuit = circuit
        self.eps = eps
        self.register_buffer('sector', torch.tensor(sector_ids()), persistent=False)
        classical = mode == MODES[2]
        self.prepare = nn.Linear(1028, 15 if classical else 12)
        self.effect = nn.Parameter(torch.empty(40, 15) if classical else torch.empty(40, 2, 3, 2))
        nn.init.normal_(self.effect, std=.2)
        self.value = nn.Linear(1024, 64)
        self.decoder = nn.Parameter(torch.empty(40, 64))
        nn.init.normal_(self.decoder, std=64**-.5)
        self.bias = nn.Parameter(torch.zeros(40))

    @staticmethod
    def real_state(coordinates):
        # Independent trainable real map, no circuit, complex dtype or Q outputs.
        squared_norm = coordinates.square().sum(dim=-1, keepdim=True)
        return torch.cat((2*coordinates, 1-squared_norm), dim=-1)/(1+squared_norm)

    def masses(self, h, roles, dephase):
        x = self.prepare(torch.cat((h, roles), dim=-1))
        if self.mode == MODES[2]:
            word_u, word_v = self.real_state(x).split(8, dim=-1)
            query_u, query_v = self.real_state(self.effect).split(8, dim=-1)
            real_blocks = []
            imag_blocks = []
            for s in (0, 1):
                keep = (self.sector == s).to(h.dtype)
                real_blocks.append((word_u*keep) @ query_u.T + (word_v*keep) @ query_v.T)
                imag_blocks.append((word_v*keep) @ query_u.T - (word_u*keep) @ query_v.T)
            block_mass = sum(re.square()+im.square() for re,im in zip(real_blocks,imag_blocks))
            presence = sum(re[..., 0] for re in real_blocks).square() + sum(im[..., 0] for im in imag_blocks).square()
        else:
            word = self.circuit(math.pi*torch.tanh(x).reshape(*x.shape[:-1], 2, 3, 2))
            query = self.circuit(math.pi*torch.tanh(self.effect))
            blocks = [(word*(self.sector == s)) @ query.T.conj() for s in (0, 1)]
            block_mass = sum(z.abs().square() for z in blocks)
            presence = block_mass[..., 0] if dephase or self.mode == MODES[1] else sum(z[..., 0] for z in blocks).abs().square()
        return .25+.5*presence, block_mass[..., 1:]

    def forward(self, h, roles, mask, details=False, dephase=False):
        if h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != (*h.shape[:2], 4):
            raise ValueError('Expected cached h and role features')
        if mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError('Each record needs at least one valid token')
        if dephase and self.mode != MODES[0]:
            raise ValueError('Frozen intervention applies to Q only')
        h = h.masked_fill(~mask[..., None], 0.)
        roles = roles.masked_fill(~mask[..., None], 0.)
        h = h.to(self.prepare.weight)
        roles = roles.to(h)
        m, a = self.masses(h, roles, dephase)
        raw = torch.cat((m[..., None], (1-m)[..., None], a), dim=-1)
        raw = (raw+self.eps)*mask[..., None]
        alpha = raw/raw.sum(dim=1, keepdim=True)
        v = self.value(h)
        pooled = alpha.transpose(1, 2) @ v
        g = (pooled[:, 0]-pooled[:, 1]) @ self.decoder[0] + self.bias[0]
        u = (pooled[:, 2:]*self.decoder[1:]).sum(dim=-1) + self.bias[1:]
        logits = torch.cat((torch.zeros_like(g[:, None]), g[:, None]+u.log_softmax(dim=-1)), dim=-1)
        if details:
            return {'logits':logits, 'presence_logodds':g, 'positive_scores':u,
                    'presence_pool':pooled[:, :2], 'identity_pool':pooled[:, 2:], 'attention':alpha}
        return logits

    def same_state_block_dephased_logits(self, h, roles, mask):
        return self(h, roles, mask, dephase=True)


def make_models(circuit, initial, seed=13):
    """Initial state is supplied by the accepted runner; no hidden seed or I/O."""
    result = {}
    with torch.random.fork_rng(devices=[]):
        for mode in MODES:
            torch.manual_seed(seed)
            model = PresenceHead(mode, circuit)
            state = model.state_dict()
            for k in (COMMON if mode == MODES[2] else tuple(state)):
                state[k] = initial[k].detach().clone()
            model.load_state_dict(state, strict=True)
            result[mode] = model
    return result
