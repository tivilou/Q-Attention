"""Uniform utility warm-up then router-only learning; stable trainer untouched."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import EPS, circuit
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum', 'dephased', 'gaussian_base')
SHARED = ('value.weight', 'value.bias', 'decoder', 'bias')
WARMUP = 256


def utility_hash(model):
    h = hashlib.sha256()
    for name in SHARED:
        h.update(name.encode())
        h.update(model.head.state_dict()[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SharedUtilityHead(Head):
    def __init__(self, kind):
        if kind not in MODES:
            raise ValueError(kind)
        super().__init__('real12' if kind == 'gaussian_base' else kind)
        self.kind = kind
        self.register_buffer('step_index', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('boundary_fingerprint', torch.zeros(32, dtype=torch.uint8))

    def set_training_progress(self, step, total):
        if total != 1024 or not 1 <= step <= total:
            raise ValueError('Fixed 256+768 staged profile')
        self.step_index.fill_(step)
        if step == WARMUP + 1:
            self.boundary_fingerprint.copy_(torch.tensor(list(bytes.fromhex(utility_hash(self))),
                dtype=torch.uint8, device=self.boundary_fingerprint.device))

    def parameter_groups(self):
        shared = dict(self.head.named_parameters())
        return ([shared[n] for n in SHARED], [p for n, p in shared.items() if n not in SHARED])

    def attention(self, h, roles, mask):
        a = self.head
        if self.kind == 'gaussian_base':
            key = a.key(torch.cat((h, roles), -1))
            scale = (key.new_tensor([3-math.sqrt(2), 3+math.sqrt(2)] * 6) / 24).sqrt()
            probability = torch.exp(-((key[:, None] - a.query[None, :, None]) * scale).square().sum(-1))
        else:
            angles = math.pi * torch.tanh(a.prepare(torch.cat((h, roles), -1))).reshape(h.shape[:2] + (2, 3, 2))
            psi = circuit(angles)
            phi = circuit(math.pi * torch.tanh(a.effect))
            probability = (torch.einsum('rd,btd->brt', phi.conj(), psi).abs().square()
                if self.kind == 'quantum' else
                torch.einsum('rd,btd->brt', phi.abs().square(), psi.abs().square()))
        raw = (probability + EPS) * mask[:, None]
        return raw / raw.sum(-1, keepdim=True)

    def forward(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Bad native frozen-word inputs')
        a = self.head
        shared, router = self.parameter_groups()
        if self.training and int(self.step_index) <= WARMUP:
            alpha = mask[:, None] / mask.sum(-1)[:, None, None]
            values, decoder, bias = a.value(h), a.decoder, a.bias
            zero = sum(p.sum() * 0 for p in router)
        else:
            alpha = self.attention(h, roles, mask)
            values = F.linear(h, a.value.weight.detach(), a.value.bias.detach())
            decoder, bias = a.decoder.detach(), a.bias.detach()
            zero = sum(p.sum() * 0 for p in shared)
        return (torch.einsum('brt,btd->brd', alpha, values) * decoder).sum(-1) + bias + zero


class StagedAdamW(torch.optim.AdamW):
    def __init__(self, model, cfg):
        self.model = model
        shared, router = model.parameter_groups()
        super().__init__([dict(params=shared), dict(params=router)], lr=cfg['lr'],
            betas=tuple(cfg['betas']), eps=cfg['optimizer_eps'], weight_decay=cfg['weight_decay'])
        self.base_lr = cfg['lr']
        self.router_reset = False

    def step(self, closure=None):
        stage2 = int(self.model.step_index) > WARMUP
        if stage2 and not self.router_reset:
            for p in self.param_groups[1]['params']:
                self.state.pop(p, None)
            self.router_reset = True
        self.param_groups[0]['lr'] = 0. if stage2 else self.base_lr
        self.param_groups[1]['lr'] = self.base_lr if stage2 else 0.
        return super().step(closure)


def make_optimizer(model, cfg):
    return StagedAdamW(model, cfg)


def make_head(kind):
    return SharedUtilityHead(kind)


def initialize(initial, seed):
    models = {}
    for kind in MODES:
        m = make_head(kind)
        native = initialize_head('real12' if kind == 'gaussian_base' else kind,
                                 initial, dict(seed=seed, training_dtype='float32'))
        m.head.load_state_dict(native.head.state_dict())
        models[kind] = m
    return models
