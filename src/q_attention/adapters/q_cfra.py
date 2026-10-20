"""Isolated second-layer intervention; unchanged base model remains available."""
from __future__ import annotations

import torch
from torch import nn
from q_attention.models.relation_transformer import AttentionInterventionPassThrough
from q_attention.plugins.q_cfra import CounterfactualFidelityAttention, MODES


def neutralize(ids, mask, subject, obj, neutral_ids, *, permuted=False):
    if ids.ndim != 2 or mask.shape != ids.shape or subject.shape != ids.shape or obj.shape != ids.shape:
        raise ValueError('unaligned Q-CFRA inputs')
    if not mask.bool().any(-1).all() or (subject.bool() & obj.bool()).any():
        raise ValueError('empty input or overlapping entity spans')
    if ((subject.bool()|obj.bool()) & ~mask.bool()).any() or not subject.any(-1).all() or not obj.any(-1).all():
        raise ValueError('entity spans outside support')
    if neutral_ids.shape != (ids.shape[0],2):
        raise ValueError('need per-row subject/object type IDs')
    if permuted:
        n = mask.sum(-1)
        positions = torch.arange(ids.shape[-1],device=ids.device)[None]
        previous = torch.where(mask.bool(),(positions-1) % n[:,None],positions)
        subject,obj = subject.gather(1,previous),obj.gather(1,previous)
    s = torch.where(subject.bool(),neutral_ids[:,0,None],ids)
    o = torch.where(obj.bool(),neutral_ids[:,1,None],ids)
    return torch.stack((ids,s,o),1)


class _CFRAHook(AttentionInterventionPassThrough):
    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self,'owner',owner)

    def after_scores(self,scores,query,key,value,*,attention_mask=None,layer_index=0):
        owner = self.owner
        if owner.scope is None:
            raise RuntimeError('Q-CFRA hook outside scoped view forward')
        hidden,mask,capture = owner.scope
        b,views,n,_ = hidden.shape
        original = scores.reshape(b,views,*scores.shape[1:])
        adjusted = original
        if owner.kernel is not None:
            geometry = owner.kernel.geometry(hidden,mask,capture)
            adjusted = owner.kernel(original,geometry,capture=capture)
        weights = adjusted.masked_fill(~mask[:,None,None,None,:].bool(),torch.finfo(scores.dtype).min).softmax(-1)
        owner.attention = weights
        if capture:
            owner.observed.update(query=query.reshape(b,views,*query.shape[1:]),
                key=key.reshape(b,views,*key.shape[1:]),value=value.reshape(b,views,*value.shape[1:]),
                base_scores=original,attention_logits=adjusted,attention=weights,
                context=weights @ value.reshape(b,views,*value.shape[1:]))
            if owner.kernel is not None:
                owner.observed.update({f'plugin_{name}':tensor for name,tensor in owner.kernel.last.items()})
        return adjusted.reshape_as(scores),value


class CFRAModel(nn.Module):
    def __init__(self,base,mode='quantum',channel=None):
        super().__init__()
        if mode not in MODES or base.config.num_layers != 2 or base.config.dropout != 0:
            raise ValueError('Q-CFRA screen requires the frozen two-layer dropout-free base')
        if any(type(layer.attn.attention_intervention) is not AttentionInterventionPassThrough for layer in base.encoder.layers):
            raise ValueError('refuse stacked interventions')
        self.base,self.mode = base,mode
        base.requires_grad_(False)
        base.classifier.requires_grad_(True)
        self.kernel = None if mode == 'disabled' else CounterfactualFidelityAttention(base.config.dim,mode,**(channel or {}))
        self.scope,self.observed,self.attention = None,{},None
        base.encoder.layers[1].attn.attention_intervention = _CFRAHook(self)

    def forward(self,input_ids,attention_mask,subject_mask,object_mask,neutral_ids,*,capture=False):
        if self.scope is not None:
            raise RuntimeError('Q-CFRA model is not reentrant')
        views = neutralize(input_ids,attention_mask,subject_mask,object_mask,neutral_ids,
                           permuted=self.mode == 'entity_permuted')
        b,v,n = views.shape
        mask = attention_mask[:,None].expand(b,v,n).reshape(b*v,n)
        subject = subject_mask[:,None].expand(b,v,n).reshape(b*v,n)
        obj = object_mask[:,None].expand(b,v,n).reshape(b*v,n)
        positions = torch.arange(n,device=input_ids.device)[None].expand(b*v,n)
        self.observed = {}
        with torch.no_grad():
            token = self.base.encoder.token_embedding(views.reshape(b*v,n))
            position = self.base.encoder.position_embedding(positions)
            embedding = token+position
            first = self.base.encoder.layers[0](embedding,mask,layer_index=0)
        self.scope = (first.reshape(b,v,n,-1),attention_mask,capture)
        try:
            hidden = self.base.encoder.layers[1](first,mask,layer_index=1)
            pooled = torch.cat([self.base._masked_mean(hidden,span) for span in (subject,obj,mask)],-1)
            logits = self.base.classifier(pooled).reshape(b,v,-1)
            if capture:
                self.observed.update(view_ids=views,attention_mask=attention_mask,subject_mask=subject_mask,
                    object_mask=object_mask,neutral_ids=neutral_ids,token_embedding=token.reshape(b,v,n,-1),
                    position_embedding=position.reshape(b,v,n,-1),embedding=embedding.reshape(b,v,n,-1),
                    first_hidden=first.reshape(b,v,n,-1),hidden=hidden.reshape(b,v,n,-1),
                    pooled=pooled.reshape(b,v,-1),logits=logits,probabilities=logits.softmax(-1))
            return logits
        finally:
            self.scope = None


def objective(logits,attention,mask,gold,*,prediction_js=.05,attention_js=.05):
    logp = logits.log_softmax(-1)
    probs = logp.exp()
    center = probs.mean(1,keepdim=True).clamp_min(1e-12).log()
    prediction = (probs*(logp-center)).sum(-1).mean()
    center_a = attention.mean(1,keepdim=True).clamp_min(1e-12).log()
    per_query = (attention*(attention.clamp_min(1e-12).log()-center_a)).sum(-1)
    support = mask[:,None,None,:].expand_as(per_query)
    consistency = (per_query*support).sum()/support.sum()
    ce = nn.functional.cross_entropy(logits[:,0],gold)
    total = ce+prediction_js*prediction+attention_js*consistency
    return total,torch.stack((ce,prediction,consistency,total))
