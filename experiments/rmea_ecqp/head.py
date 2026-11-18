"""Entity-conditioned preparation and explicitly matched feature controls.

The primary route keeps the original circuit, Born response, EPS and signed
readout. Classical routes use the same twelve pre-circuit features and number
of parameters; their computation and induced function classes differ.
"""
import hashlib
import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead, circuit

ROUTINGS = ('quantum', 'softmax12', 'real_overlap12')
FEATURES = ('entity', 'token_square', 'token_linear')
SIGN_SHA256 = 'e028c261fb310a727de4a1929d1b704d98555b051e8f83b99e4e8a5394ca659d'


def fixed_projection(dtype=torch.float32, device=None):
    """A fixed buffer, independent of labels, features and Torch RNG state."""
    values = [1 if hashlib.sha256(f'ecqp-projection-v1:{i}:{j}'.encode()).digest()[0] & 1 else -1
        for i in range(16) for j in range(1024)]
    return torch.tensor(values, dtype=dtype, device=device).reshape(16, 1024) / 32


def projection_sign_fingerprint(projection):
    p = projection.detach().cpu()
    if p.shape != (16, 1024) or not torch.isfinite(p).all() or not ((p == 1 / 32) | (p == -1 / 32)).all():
        raise ValueError('Invalid fixed projection entries')
    raw = bytes(1 if v > 0 else 255 for v in p.reshape(-1).tolist())
    return hashlib.sha256(raw).hexdigest()


def cohort_derangement(ids, groups, split):
    """Whole-cohort, label-blind donors from a single already-separated split.

    Groups must be frozen from input entity identities before model outputs.
    Contiguous group blocks, shifted by the largest block, avoid every group
    when its size is at most half the cohort. Infeasible cohorts are rejected.
    The caller persists this mapping; it is never recomputed per minibatch.
    """
    if not isinstance(split, str) or not split or len(ids) != len(groups) or len(ids) < 2:
        raise ValueError('One explicit split and a nontrivial whole cohort are required')
    if any(not isinstance(x, str) or not x for x in [*ids, *groups]) or len(set(ids)) != len(ids):
        raise ValueError('Unique string sample IDs and nonempty input-derived groups are required')
    grouped = defaultdict(list)
    for sample, group in zip(ids, groups):
        grouped[group].append(sample)
    size = max(map(len, grouped.values()))
    if 2 * size > len(ids):
        raise ValueError('No group-excluding bijective donor map exists for this cohort')
    def order(kind, value):
        return hashlib.sha256(f'ecqp-donor-{kind}-v1:{split}:{value}'.encode()).digest(), value
    ordered = [sample for group in sorted(grouped, key=lambda g: order('group', g))
        for sample in sorted(grouped[group], key=lambda s: order('id', s))]
    donor = {sample: ordered[(i + size) % len(ordered)] for i, sample in enumerate(ordered)}
    by_id = dict(zip(ids, groups))
    if set(donor) != set(donor.values()) or any(by_id[s] == by_id[d] for s, d in donor.items()):
        raise RuntimeError('Donor construction invariant failed')
    return donor


class ECQPHead(RMEAHead):
    """The exploratory head; every feature/routing choice is an explicit control."""
    def __init__(self, routing='quantum', feature='entity'):
        if routing not in ROUTINGS or feature not in FEATURES:
            raise ValueError('Unknown ECQP routing or feature control')
        super().__init__('quantum')
        self.routing = routing
        self.feature = feature
        self.mode = 'ecqp_' + routing
        # No new random draw: the common legacy initialization is preserved.
        self.B = nn.Parameter(torch.zeros(12, 16))
        self.register_buffer('projection', fixed_projection(), persistent=True)
        self.verify_projection()

    def verify_projection(self):
        if projection_sign_fingerprint(self.projection) != SIGN_SHA256:
            raise ValueError('Projection identity differs from the frozen rule')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'projection' in state_dict and projection_sign_fingerprint(state_dict['projection']) != SIGN_SHA256:
            raise ValueError('Checkpoint projection identity differs from the frozen rule')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.verify_projection()
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # nn.Module.load_state_dict recurses through this method, not through a
        # child's load_state_dict override. Missing P is an error even with
        # strict=False: a checkpoint must carry the frozen persistent identity.
        key = prefix + 'projection'
        try:
            if key not in state_dict or projection_sign_fingerprint(state_dict[key]) != SIGN_SHA256:
                raise ValueError('Missing or substituted fixed projection')
        except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
            error_msgs.append(key + ': invalid frozen projection (' + type(exc).__name__ + ')')
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # Loading is not transaction-atomic for the complete containing module;
        # callers must discard any model whose load raised an error.

    def load_native(self, native):
        """Copy only unchanged original members, without silently replacing B/P."""
        if not isinstance(native, RMEAHead) or native.mode != 'quantum':
            raise ValueError('An original quantum RMEA head is required')
        source = native.state_dict()
        destination = self.state_dict()
        if set(destination) - set(source) != {'B', 'projection'} or set(source) - set(destination):
            raise RuntimeError('Unexpected shared-state correspondence')
        self.verify_projection()
        with torch.no_grad():
            for name, value in source.items():
                if destination[name].shape != value.shape:
                    raise ValueError('Native shared-state shape mismatch')
                destination[name].copy_(value)
        self.verify_projection()

    @staticmethod
    def checked_inputs(h, roles, mask):
        if h.ndim != 3 or h.shape[-1] != 1024 or h.dtype not in (torch.float32, torch.float64):
            raise ValueError('Expected float32/float64 word features [B,T,1024]')
        if roles.shape != h.shape[:2] + (4,) or roles.dtype != h.dtype or roles.device != h.device:
            raise ValueError('Role features must share word dtype/device and shape [B,T,4]')
        if mask.shape != h.shape[:2] or mask.dtype != torch.bool or mask.device != h.device or not mask.any(-1).all():
            raise ValueError('Each sample needs a boolean valid-word mask')
        if not torch.isfinite(h[mask]).all() or not torch.isfinite(roles[mask]).all():
            raise ValueError('Valid words and roles must be finite')
        flags = roles[..., 2:4][mask]
        if not ((flags == 0) | (flags == 1)).all():
            raise ValueError('Entity-role channels must be binary')
        # Padding never contributes, including when a caller leaves NaNs there.
        h = torch.where(mask[..., None], h, torch.zeros_like(h))
        roles = torch.where(mask[..., None], roles, torch.zeros_like(roles))
        weights = roles[..., 2:4] * mask[..., None]
        counts = weights.sum(1)
        if not (counts > 0).all():
            raise ValueError('Both entities need at least one valid word; no silent fallback')
        entities = torch.einsum('bte,btd->bed', weights, h) / counts[..., None]
        return h, roles, entities

    def forward(self, h, roles, mask, details=False, anchor_override=None):
        h, roles, entities = self.checked_inputs(h, roles, mask)
        if h.dtype != self.B.dtype or h.device != self.B.device:
            raise ValueError('Head and features must share dtype/device')
        anchor = entities.mean(1)
        if anchor_override is not None:
            if self.feature != 'entity' or anchor_override.shape != anchor.shape or anchor_override.dtype != h.dtype or anchor_override.device != h.device:
                raise ValueError('Only entity control accepts a matching explicit cohort donor summary')
            if not torch.isfinite(anchor_override).all():
                raise ValueError('Donor summaries must be finite')
            anchor = anchor_override
        projected_word = F.linear(h, self.projection)
        projected_entity = F.linear(anchor, self.projection)
        if self.feature == 'entity':
            interaction = projected_word * projected_entity[:, None, :]
        elif self.feature == 'token_square':
            interaction = projected_word.square()
        else:
            interaction = projected_word
        if not torch.isfinite(projected_word[mask]).all() or not torch.isfinite(projected_entity).all() or not torch.isfinite(interaction[mask]).all():
            raise ValueError('Projected interaction overflow; no silent rescaling')
        raw = self.prepare(torch.cat((h, roles), -1)) + F.linear(interaction, self.B)
        if not torch.isfinite(raw[mask]).all():
            raise ValueError('Preparation overflow; no silent clipping')
        angles = math.pi * torch.tanh(raw)
        effect_angles = math.pi * torch.tanh(self.effect)
        psi = phi = None
        if self.routing == 'quantum':
            psi = circuit(angles.reshape(h.shape[:2] + (2, 3, 2)))
            phi = circuit(effect_angles)
            probability = torch.einsum('rd,btd->brt', phi.conj(), psi).abs().square()
            numerator = (probability + EPS) * mask[:, None, :]
            normalizer = numerator.sum(-1, keepdim=True)
            alpha = numerator / normalizer
            scores = probability
        else:
            query = effect_angles.reshape(40, 12)
            if self.routing == 'softmax12':
                scores = torch.einsum('rd,btd->brt', query, angles) / math.sqrt(12)
                alpha = scores.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
                probability = normalizer = None
            else:
                scores = torch.einsum('rd,btd->brt', F.normalize(query, dim=-1), F.normalize(angles, dim=-1)).square()
                probability = scores
                numerator = (scores + EPS) * mask[:, None, :]
                normalizer = numerator.sum(-1, keepdim=True)
                alpha = numerator / normalizer
        values = self.value(h)
        utilities = torch.einsum('btd,rd->brt', values, self.decoder)
        logits = (alpha * utilities).sum(-1) + self.bias
        if not details:
            return logits, alpha
        return dict(logits=logits, alpha=alpha, values=values, utilities=utilities,
            prepare_raw=raw, angles=angles.reshape(h.shape[:2] + (2, 3, 2)),
            effect_angles=effect_angles, psi=psi, phi=phi, probability=probability,
            normalizer=normalizer, routing_scores=scores, entities=entities,
            entity_summary=anchor, projected_word=projected_word,
            projected_entity=projected_entity, interaction=interaction,
            routing=self.routing, feature=self.feature)
