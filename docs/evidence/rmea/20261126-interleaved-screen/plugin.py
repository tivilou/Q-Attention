"""Middle relation operator. Scientific head only; native parameter schema.

Q: |<0|W2 Q^dag W1|0>|^2. D dephases at those three block
boundaries, preserving interference *within* Q. real8 uses two RY-ring
layers per word segment and four per relation; gate geometry differs.
"""
import math
import torch
from torch import nn
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head

MODES = ('quantum_middle', 'dephased_middle', 'real8_middle')
SHARED = ('value.weight', 'value.bias', 'decoder', 'bias')


def layer(angles, state):
    """One native RY/RZ/ring block, wire0 MSB; column-state action."""
    for wire in range(3):
        shape = state.shape
        pair = state.reshape(shape[:-1] + (2**wire, 2, 2**(2-wire)))
        a, b = pair.unbind(-2)
        y, z = angles[..., wire, :].unbind(-1)
        c, s = torch.cos(y/2)[..., None, None], torch.sin(y/2)[..., None, None]
        minus, plus = torch.exp(-.5j*z)[..., None, None], torch.exp(.5j*z)[..., None, None]
        state = torch.stack(((c*a-s*b)*minus, (s*a+c*b)*plus), -2).reshape(shape)
    for control, target in ((0, 1), (1, 2), (2, 0)):
        perm = [i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
        state = state[..., perm]
    return state


def layer_matrix(angles):
    dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
    eye = torch.eye(8, device=angles.device, dtype=dtype).expand(angles.shape[:-2]+(8, 8))
    return layer(angles[..., None, :, :], eye).transpose(-1, -2)


def phase_free(angles):
    return torch.stack((angles[..., 0], torch.zeros_like(angles[..., 1])), -1)


def terminal_bra(y):
    """Native <0|W2 up to a scalar global phase; no final RZ dependence."""
    out = torch.ones(y.shape[:-1]+(1,), dtype=y.dtype, device=y.device)
    for wire in range(3):
        pair = torch.stack((torch.cos(y[..., wire]/2), -torch.sin(y[..., wire]/2)), -1)
        out = (out[..., :, None]*pair[..., None, :]).flatten(-2)
    return out


def ry_matrix(angles):
    """Ordered RY+ring layers. Entire matrix; inverse is true transpose."""
    eye = torch.eye(8, dtype=angles.dtype, device=angles.device)
    state = eye.expand(angles.shape[:-2]+(8, 8))
    for depth in range(angles.shape[-2]):
        for wire in range(3):
            shape = state.shape
            pair = state.reshape(shape[:-1]+(2**wire, 2, 2**(2-wire)))
            a, b = pair.unbind(-2)
            y = angles[..., depth, wire, None, None, None]
            c, s = torch.cos(y/2), torch.sin(y/2)
            state = torch.stack((c*a-s*b, s*a+c*b), -2).reshape(shape)
        for control, target in ((0, 1), (1, 2), (2, 0)):
            perm = [i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            state = state[..., perm]
    return state.transpose(-1, -2)


def probability(word, relation, kind):
    """Angles already transformed by native pi*tanh; no task values here."""
    if kind not in MODES or word.shape[-3:] != (2, 3, 2) or relation.shape != (40, 2, 3, 2):
        raise ValueError('Bad middle-placement angles or mode')
    if kind == 'real8_middle':
        flat = word.flatten(-3)
        w1 = ry_matrix(flat[..., :6].reshape(word.shape[:-3]+(2, 3)))
        w2 = ry_matrix(flat[..., 6:].reshape(word.shape[:-3]+(2, 3)))
        q = ry_matrix(relation.reshape(40, 4, 3))
        amplitude = torch.einsum('bti,rij,btj->brt', w2[..., 0, :], q.transpose(-1, -2), w1[..., :, 0])
        return amplitude.square()
    bra = terminal_bra(word[..., 1, :, 0])
    first = word[..., 0, :, :]
    second_query = relation[:, 1]
    if kind == 'dephased_middle':
        # Both discarded phase sets are analytically invisible. Keep their
        # registered tensors and exact zero route gradients (AdamW still decays).
        first = phase_free(first)
        second_query = phase_free(second_query)
    dtype = torch.complex128 if word.dtype == torch.float64 else torch.complex64
    zero = torch.zeros(word.shape[:-3]+(8,), dtype=dtype, device=word.device)
    zero[..., 0] = 1
    b = layer(first, zero)
    q = layer_matrix(second_query) @ layer_matrix(relation[:, 0])
    qdag = q.conj().transpose(-1, -2)
    if kind == 'dephased_middle':
        return torch.einsum('bti,rij,btj->brt', bra.square(), qdag.abs().square(), b.abs().square())
    amplitude = torch.einsum('bti,rij,btj->brt', bra.to(dtype), qdag, b)
    return amplitude.abs().square()


class MiddleHead(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in MODES:
            raise ValueError(kind)
        self.kind = kind
        self.head = Head('quantum').head

    def components(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid shared frozen-word inputs')
        a = self.head
        word = (math.pi*torch.tanh(a.prepare(torch.cat((h, roles), -1)))).reshape(h.shape[:2]+(2, 3, 2))
        relation = math.pi*torch.tanh(a.effect)
        p = probability(word, relation, self.kind)
        raw = (p+EPS)*mask[:, None]
        alpha = raw/raw.sum(-1, keepdim=True)
        utilities = torch.einsum('btd,rd->brt', a.value(h), a.decoder)
        return p, alpha, utilities

    def forward(self, h, roles, mask):
        _, alpha, utilities = self.components(h, roles, mask)
        return (alpha*utilities).sum(-1)+self.head.bias


def make_head(kind):
    return MiddleHead(kind)


def initialize(initial, seed):
    dtype = next(iter(initial.values())).dtype
    models = {}
    for kind in MODES:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = MiddleHead(kind).to(dtype=dtype)
        model.head.load_state_dict(initial, strict=True)
        if any(not torch.equal(model.head.state_dict()[k], v) for k, v in initial.items()):
            raise ValueError('Initial tensor mapping changed')
        models[kind] = model
    return models
