"""Three-register relation measurements over frozen original-word features.

This is a bounded-development candidate, not an accepted experiment protocol.
Wire zero is MSB. Each effect is a separate binary rank-one measurement; the
forty effects are not a POVM. No quantum computational advantage is claimed.
"""
import math

import torch
from torch import nn

from .frozen_feature_head import FrozenFeatureHead
from .rmea import EPS, RMEAHead
from .rmea_entity import entity_inputs

VARIANTS = ('static', 'joint', 'no_content', 'separable', 'classical_cp')
LOCAL_EDGES = ((0, 1), (2, 3), (4, 5))
CROSS_EDGES = ((0, 2), (2, 4), (4, 0))


def register_state(angles, edges):
    """One RY/RZ layer followed by the ordered CNOT edges, from |0...0>."""
    if angles.ndim < 2 or angles.shape[-1] != 2 or angles.dtype not in (torch.float32, torch.float64):
        raise ValueError('Expected real [...,n_wires,2] angles')
    wires = angles.shape[-2]
    if wires not in (2, 6):
        raise ValueError('Only two- and six-wire registers are defined')
    dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
    shape = angles.shape[:-2]
    state = torch.zeros(shape + (2**wires,), device=angles.device, dtype=dtype)
    state[..., 0] = 1
    for wire in range(wires):
        pair = state.reshape(shape + (2**wire, 2, 2**(wires-wire-1)))
        a, b = pair.unbind(-2)
        y, z = angles[..., wire, :].unbind(-1)
        c, s = torch.cos(y/2)[..., None, None], torch.sin(y/2)[..., None, None]
        minus, plus = torch.exp(-.5j*z)[..., None, None], torch.exp(.5j*z)[..., None, None]
        state = torch.stack(((c*a-s*b)*minus, (s*a+c*b)*plus), -2).reshape(shape + (2**wires,))
    for control, target in edges:
        if not 0 <= control < wires or not 0 <= target < wires or control == target:
            raise ValueError('Invalid CNOT edge')
        perm = [i ^ ((1 << (wires-1-target)) if (i >> (wires-1-control)) & 1 else 0)
                for i in range(2**wires)]
        state = state[..., perm]
    return state


def normalized_attention(probability, mask, epsilon=EPS):
    """Smooth nonnegative scores then mask. Reject empty rows, never fill them."""
    if epsilon <= 0 or mask.dtype != torch.bool or probability.ndim != 3:
        raise ValueError('Positive smoothing, scores[B,R,T], bool mask[B,T] required')
    if mask.shape != (probability.shape[0], probability.shape[2]) or not mask.any(-1).all():
        raise ValueError('Invalid valid-word mask')
    raw = (probability + epsilon) * mask[:, None, :]
    total = raw.sum(-1, keepdim=True)
    return raw / total, total


def separable_floor(q, c, mask, epsilon=EPS):
    """Exact product-effect null: alpha=lambda*q/Q+(1-lambda)*uniform.

    q[B,R,T] is token overlap squared, c[B,R,1] the entity overlap product.
    Q=0 has lambda=0 and a defined uniform limit, without 0/0 or gradient NaNs.
    """
    if epsilon <= 0 or c.shape != q.shape[:2] + (1,):
        raise ValueError('Bad separable floor inputs')
    valid_q = q * mask[:, None, :]
    mass = valid_q.sum(-1, keepdim=True)
    length = mask.sum(-1)[:, None, None].to(q.dtype)
    if (length == 0).any():
        raise ValueError('Empty row')
    direction = valid_q / torch.where(mass > 0, mass, torch.ones_like(mass))
    mixture = c * mass / (c * mass + length * epsilon)
    uniform = mask[:, None, :].to(q.dtype) / length
    return dict(alpha=mixture*direction+(1-mixture)*uniform, q=q, c=c,
                q_mass=mass, direction=direction, mixture=mixture)


class JointRMEAHead(FrozenFeatureHead):
    """Explicit registry. Same value/decoder initialization as legacy RMEA.

    joint/no_content/separable: 81,028 parameters. classical_cp: 81,508,
    deliberately a flexible replacement, NOT exact capacity equivalence.
    separable has structurally redundant entity gradients as epsilon -> 0;
    nominal parameter equality must not be called equal active capacity.
    """
    def __init__(self, variant):
        if variant not in VARIANTS or variant == 'static':
            raise ValueError('Use make_head for the exact static adapter')
        super().__init__('uniform')
        self.variant = variant
        self.prepare_word = nn.Linear(1028, 4)
        self.prepare_subject = nn.Linear(1028, 4)
        self.prepare_object = nn.Linear(1028, 4)
        if variant == 'classical_cp':
            self.cp = nn.Parameter(torch.randn(40, 2, 3, 4) * .2)
        else:
            self.effect = nn.Parameter(torch.randn(40, 6, 2) * .2)
        self.register_buffer('role_means', torch.zeros(2, 1028))
        self.register_buffer('role_mean_count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def fit_role_means(self, batches, *, split):
        if self.variant != 'no_content' or split != 'train' or self.role_mean_count.item():
            raise ValueError('Only no_content may calibrate once on train')
        total, count = torch.zeros(2, 1028, dtype=torch.float64), 0
        for h, roles, mask in batches:
            pair = entity_inputs(h, roles, mask).detach().to(device='cpu', dtype=torch.float64)
            total += pair.sum(0)
            count += pair.shape[0]
        if not count:
            raise ValueError('Empty training calibration')
        self.role_means.copy_((total/count).to(self.role_means))
        self.role_mean_count.fill_(count)

    def forward(self, h, roles, mask, details=False, *, entity_override=None):
        pair = entity_inputs(h, roles, mask)
        if self.variant == 'no_content':
            if entity_override is not None or not self.role_mean_count.item():
                raise ValueError('Train means required; override forbidden for no_content')
            pair = self.role_means[None].expand_as(pair)
        elif entity_override is not None:
            if entity_override.shape != pair.shape or entity_override.dtype != pair.dtype or entity_override.device != pair.device:
                raise ValueError('Entity override shape/dtype/device mismatch')
            if not torch.isfinite(entity_override).all():
                raise ValueError('Nonfinite entity override')
            pair = entity_override
        word_angles = math.pi * torch.tanh(self.prepare_word(torch.cat((h, roles), -1)))
        subject_angles = math.pi * torch.tanh(self.prepare_subject(pair[:, 0]))
        object_angles = math.pi * torch.tanh(self.prepare_object(pair[:, 1]))
        diagnostic = dict(word_angles=word_angles, subject_angles=subject_angles,
                          object_angles=object_angles, entity_inputs=pair)
        if self.variant == 'classical_cp':
            # Same bounded four-angle inputs; a real rank-two trilinear amplitude.
            w = torch.einsum('btd,rkd->brkt', word_angles, self.cp[:, :, 0])
            s = torch.einsum('bd,rkd->brk', subject_angles, self.cp[:, :, 1])
            o = torch.einsum('bd,rkd->brk', object_angles, self.cp[:, :, 2])
            amplitude = (w * (s*o)[..., None]).sum(2)
            probability = amplitude.square()
            diagnostic.update(amplitude=amplitude, cp=self.cp)
        else:
            psi = register_state(word_angles.reshape(h.shape[:2] + (2, 2)), ((0, 1),))
            subject = register_state(subject_angles.reshape(-1, 2, 2), ((0, 1),))
            obj = register_state(object_angles.reshape(-1, 2, 2), ((0, 1),))
            effect_angles = math.pi * torch.tanh(self.effect)
            edges = LOCAL_EDGES + (() if self.variant == 'separable' else CROSS_EDGES)
            phi = register_state(effect_angles, edges)
            # Contract entities first. No B*T*R*64 tensor, no density matrices,
            # and crucially NO normalization of this conditional vector.
            conditional = torch.einsum('rijk,bj,bk->bri', phi.conj().reshape(40, 4, 4, 4), subject, obj)
            amplitude = torch.einsum('bri,bti->brt', conditional, psi)
            probability = amplitude.abs().square()
            diagnostic.update(psi=psi, subject=subject, object=obj, phi=phi,
                              effect_angles=effect_angles, conditional=conditional,
                              amplitude=amplitude)
            if self.variant == 'separable':
                a = register_state(effect_angles[:, :2], ((0, 1),))
                b = register_state(effect_angles[:, 2:4], ((0, 1),))
                c = register_state(effect_angles[:, 4:], ((0, 1),))
                q = torch.einsum('ri,bti->brt', a.conj(), psi).abs().square()
                factor = (torch.einsum('ri,bi->br', b.conj(), subject).abs().square()
                          * torch.einsum('ri,bi->br', c.conj(), obj).abs().square())[..., None]
                diagnostic['separable_null'] = separable_floor(q, factor, mask)
        alpha, normalizer = normalized_attention(probability, mask)
        values = self.value(h)
        utilities = torch.einsum('btd,rd->brt', values, self.decoder)
        logits = (alpha * utilities).sum(-1) + self.bias
        if not details:
            return logits, alpha
        diagnostic.update(probability=probability, alpha=alpha, normalizer=normalizer,
                          values=values, utilities=utilities, logits=logits)
        return diagnostic


def make_head(variant):
    if variant == 'static':
        return RMEAHead('quantum')
    return JointRMEAHead(variant)
