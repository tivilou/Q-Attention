"""Ordered entity-conditioned RMEA; legacy RMEA remains unchanged.

This module accepts frozen word features, never labels. Binary relation effects
are separate measurements, not a joint POVM. No hardware advantage is implied.
"""
import math

import torch

from .rmea import DEPTH, EPS, QUBITS, RMEAHead, circuit


VARIANTS = ('static', 'ordered', 'no_content', 'symmetric', 'classical_pair')


def apply_circuit(angles, state, inverse=False):
    """Apply U or U† to arbitrary complex states with broadcast batch axes.

    Circuit order matches legacy circuit: RY then RZ on each wire, followed
    by 0->1,1->2,2->0 CNOTs, repeated twice. Wire zero is most significant.
    """
    if angles.shape[-3:] != (DEPTH, QUBITS, 2) or state.shape[-1] != 8:
        raise ValueError('Expected [...,2,3,2] angles and [...,8] states')
    if angles.dtype not in (torch.float32, torch.float64):
        raise ValueError('Angles must be float32 or float64')
    expected = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
    if state.dtype != expected or state.device != angles.device:
        raise ValueError('State dtype/device must match angle precision/device')
    shape = torch.broadcast_shapes(angles.shape[:-3], state.shape[:-1])
    angles = angles.expand(shape + (DEPTH, QUBITS, 2))
    state = state.expand(shape + (8,))

    def rotate(value, wire, angle, axis):
        pair = value.reshape(shape + (2**wire, 2, 2**(2-wire)))
        a, b = pair.unbind(-2)
        if axis == 'y':
            c, s = torch.cos(angle/2)[..., None, None], torch.sin(angle/2)[..., None, None]
            out = (c*a-s*b, s*a+c*b)
        else:
            out = (a*torch.exp(-.5j*angle)[..., None, None],
                   b*torch.exp(.5j*angle)[..., None, None])
        return torch.stack(out, -2).reshape(shape + (8,))

    def entangle(value, control, target):
        perm = [i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
        return value[..., perm]

    layers = range(DEPTH-1, -1, -1) if inverse else range(DEPTH)
    ring = ((0, 1), (1, 2), (2, 0))
    for layer in layers:
        if inverse:
            for control, target in reversed(ring):
                state = entangle(state, control, target)
            for wire in range(QUBITS-1, -1, -1):
                state = rotate(state, wire, -angles[..., layer, wire, 1], 'z')
                state = rotate(state, wire, -angles[..., layer, wire, 0], 'y')
        else:
            for wire in range(QUBITS):
                state = rotate(state, wire, angles[..., layer, wire, 0], 'y')
                state = rotate(state, wire, angles[..., layer, wire, 1], 'z')
            for control, target in ring:
                state = entangle(state, control, target)
    return state


def entity_inputs(h, roles, mask):
    """Mean of each example's valid subject/object words including role features."""
    if h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2] + (4,):
        raise ValueError('Expected h[B,T,1024] and roles[B,T,4]')
    if mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Bad valid-word mask')
    if h.dtype != roles.dtype or h.device != roles.device or mask.device != h.device:
        raise ValueError('Feature dtype/device mismatch')
    if h.dtype not in (torch.float32, torch.float64):
        raise ValueError('Head inputs must be float32 or float64')
    if not torch.isfinite(h).all() or not torch.isfinite(roles).all():
        raise ValueError('Nonfinite features')
    memberships = roles[..., 2:4]
    if not ((memberships == 0) | (memberships == 1)).all():
        raise ValueError('Entity memberships must be binary')
    membership = memberships.transpose(1, 2) * mask[:, None, :]
    counts = membership.sum(-1, keepdim=True)
    if (counts == 0).any():
        raise ValueError('Empty subject/object after valid-word masking')
    features = torch.cat((h, roles), -1)
    return torch.einsum('bet,btd->bed', membership/counts, features)


class EntityRMEAHead(RMEAHead):
    """Explicit five-variant registry with equal active trainable parameter counts.

    mode remains 'quantum' for the exact legacy static adapter; variant selects
    the new behavior. no_content requires train-only calibration before forward.
    """
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError('Unknown entity-conditioned RMEA variant')
        super().__init__('quantum')
        self.variant = variant
        self.register_buffer('role_means', torch.zeros(2, 1028))
        self.register_buffer('role_mean_count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def fit_role_means(self, batches, *, split):
        """Equal-example mean of pooled entities; deliberately accepts no labels."""
        if split != 'train' or self.variant != 'no_content':
            raise ValueError('Only no_content may calibrate on train')
        if self.role_mean_count.item():
            raise ValueError('Role means already frozen')
        total = torch.zeros(2, 1028, dtype=torch.float64)
        count = 0
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
        if self.variant == 'static':
            if entity_override is not None:
                raise ValueError('Static adapter has no entity branch')
            return super().forward(h, roles, mask, details)
        if self.variant == 'no_content':
            if not self.role_mean_count.item() or entity_override is not None:
                raise ValueError('Train-only frozen means required; overrides forbidden')
            pair = self.role_means[None].expand(h.shape[0], -1, -1)
        elif entity_override is not None:
            if (entity_override.shape != pair.shape or entity_override.dtype != h.dtype
                    or entity_override.device != h.device or not torch.isfinite(entity_override).all()):
                raise ValueError('Bad branch-only entity override')
            pair = entity_override
        features = torch.cat((h, roles), -1)
        word_raw = self.prepare(features)
        pair_raw = self.prepare(pair)
        word_angles = math.pi*torch.tanh(word_raw)
        pair_angles = math.pi*torch.tanh(pair_raw)
        a_s, a_o = pair_angles.reshape(h.shape[0], 2, DEPTH, QUBITS, 2).unbind(1)
        shared = dict(features=features, entity_pair=pair, entity_raw=pair_raw,
                      subject_angles=a_s, object_angles=a_o, prepare_raw=word_raw,
                      angles=word_angles.reshape(h.shape[:2] + (DEPTH, QUBITS, 2)))
        if self.variant == 'classical_pair':
            # Identical bounded angle mapping and number of active relation parameters.
            query = math.pi*torch.tanh(self.effect.reshape(40, 12))
            conditioned_query = query[None] + pair_angles[:, 0, None] - pair_angles[:, 1, None]
            scores = torch.einsum('brd,btd->brt', conditioned_query, word_angles)/math.sqrt(12)
            alpha = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            shared.update(query=query, conditioned_query=conditioned_query, scores=scores)
        else:
            psi = circuit(shared['angles'])
            effect_angles = math.pi*torch.tanh(self.effect)
            phi = circuit(effect_angles)
            # <phi|Uo Us|psi> == <Us† Uo† phi|psi>.
            effective = apply_circuit(a_s[:, None], apply_circuit(a_o[:, None], phi[None], True), True)
            probability = torch.einsum('brd,btd->brt', effective.conj(), psi).abs().square()
            shared.update(psi=psi, phi=phi, effect_angles=effect_angles, effective_phi=effective,
                          ordered_probability=probability)
            if self.variant == 'symmetric':
                reverse = apply_circuit(a_o[:, None], apply_circuit(a_s[:, None], phi[None], True), True)
                reversed_probability = torch.einsum('brd,btd->brt', reverse.conj(), psi).abs().square()
                probability = .5*(probability+reversed_probability)
                shared.update(reverse_effective_phi=reverse, reverse_probability=reversed_probability)
            raw = (probability+EPS)*mask[:, None]
            normalizer = raw.sum(-1, keepdim=True)
            alpha = raw/normalizer
            shared.update(probability=probability, normalizer=normalizer)
        values = self.value(h)
        utilities = torch.einsum('btd,rd->brt', values, self.decoder)
        logits = (alpha*utilities).sum(-1)+self.bias
        if not details:
            return logits, alpha
        shared.update(logits=logits, alpha=alpha, values=values, utilities=utilities,
                      decoder=self.decoder, bias=self.bias, effect_raw=self.effect)
        return shared


HEADS = {name: (lambda variant=name: EntityRMEAHead(variant)) for name in VARIANTS}


def make_head(variant):
    if variant not in HEADS:
        raise ValueError('Variant not allowlisted')
    return HEADS[variant]()


def mismatched_pairs(pair, ids):
    """Label-blind same-split cyclic derangement in stable ID-hash order.

    The caller must supply the whole selected split, never independently chunk
    a split. Singleton populations are rejected rather than called mismatches.
    """
    import hashlib
    if len(ids) != len(pair) or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('Unique IDs and at least two same-split examples required')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(str(ids[i]).encode()).digest())
    indices = list(range(len(ids)))
    for dest, source in zip(order, order[1:]+order[:1]):
        indices[dest] = source
    return pair[torch.tensor(indices, device=pair.device)]
