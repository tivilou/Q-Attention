"""Isolated Q-SRM context hook; the disabled path uses the original softmax."""
from __future__ import annotations

import torch
from torch import nn
from q_attention.models.relation_transformer import (
    AttentionContextPassThrough, AttentionInterventionPassThrough, AttentionScorePassThrough,
)
from q_attention.plugins.q_srm import MODES, SquareRootMeasurementAttention


class _SRMContext(AttentionContextPassThrough):
    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, 'owner', owner)

    def forward(self, query, key, value, scores, attention_mask):
        owner = self.owner
        if owner.scope is None:
            raise RuntimeError('Q-SRM context called outside scoped forward')
        capture = owner.scope
        if owner.kernel is None:
            attention = scores.masked_fill(~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min).softmax(-1)
            null = torch.zeros_like(attention[..., 0])
        else:
            attention, null = owner.kernel(query, key, attention_mask, capture=capture)
        # Compute the context in probability precision, then return base dtype.
        context = (attention @ value.to(attention.dtype)).to(value.dtype)
        owner.attention, owner.null_probability = attention, null
        if capture:
            owner.observed.update(query=query, key=key, value=value, base_scores=scores,
                                  attention=attention, null_probability=null, context=context)
            if owner.kernel is not None:
                owner.observed.update({'plugin_'+k: v for k, v in owner.kernel.last.items()})
        # None preserves the unmodified base path exactly, including its dropout.
        return None if owner.kernel is None else context


class SRMModel(nn.Module):
    def __init__(self, base, mode='quantum'):
        super().__init__()
        if mode not in MODES or base.config.num_layers != 2 or base.config.dropout != 0:
            raise ValueError('Q-SRM v1 requires a two-layer dropout-free base and explicit mode')
        if base.config.dim // base.config.num_heads != 8 or base.model_parallel_enabled:
            raise ValueError('Q-SRM v1 requires head_dim=8 and a single device')
        for layer in base.encoder.layers:
            if (type(layer.attn.attention_intervention) is not AttentionInterventionPassThrough
                or type(layer.attn.score_intervention) is not AttentionScorePassThrough
                or type(layer.attn.context_intervention) is not AttentionContextPassThrough):
                raise ValueError('refuse stacked interventions')
        self.base, self.mode = base, mode
        base.requires_grad_(False)
        base.classifier.requires_grad_(True)
        self.kernel = None if mode == 'disabled' else SquareRootMeasurementAttention(mode)
        self.scope, self.observed = None, {}
        self.attention, self.null_probability = None, None
        base.encoder.layers[1].attn.context_intervention = _SRMContext(self)

    def forward(self, input_ids, attention_mask, subject_mask, object_mask, *, capture=False):
        if self.scope is not None:
            raise RuntimeError('Q-SRM model is not reentrant')
        if (input_ids.ndim != 2 or any(x.shape != input_ids.shape for x in (attention_mask, subject_mask, object_mask))
            or any(x.dtype != torch.bool for x in (attention_mask, subject_mask, object_mask))):
            raise ValueError('unaligned input or nonboolean masks')
        if (not attention_mask.any(-1).all() or not subject_mask.any(-1).all() or not object_mask.any(-1).all()
            or (subject_mask & object_mask).any() or ((subject_mask | object_mask) & ~attention_mask).any()):
            raise ValueError('empty, overlapping or unsupported entity spans')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
        self.observed = {}
        with torch.no_grad():
            token = self.base.encoder.token_embedding(input_ids)
            position = self.base.encoder.position_embedding(positions)
            embedding = token + position
            first = self.base.encoder.layers[0](embedding, attention_mask, layer_index=0)
        self.scope = capture
        try:
            hidden = self.base.encoder.layers[1](first, attention_mask, layer_index=1)
            pooled = torch.cat([self.base._masked_mean(hidden, mask) for mask in (subject_mask, object_mask, attention_mask)], -1)
            logits = self.base.classifier(pooled)
            if capture:
                self.observed.update(input_ids=input_ids, attention_mask=attention_mask, subject_mask=subject_mask,
                                     object_mask=object_mask, token_embedding=token, position_embedding=position,
                                     embedding=embedding, first_hidden=first, hidden=hidden, pooled=pooled,
                                     logits=logits, probabilities=logits.softmax(-1))
            return logits
        finally:
            self.scope = None


def objective(logits, gold):
    loss = nn.functional.cross_entropy(logits, gold)
    return loss, loss.reshape(1)
