"""Multilayer observations around the unchanged relation-model forward."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from q_attention.models.relation_transformer import AttentionInterventionPassThrough
from q_attention.plugins.q_ecpa import EntityCoherentPath, ECPAConfig
from q_attention.plugins.q_query_key_coherent_transport import (
    QuantumQueryKeyCoherentTransportKernel, QueryKeyCoherentTransportConfig, _ry_matrix)


class ECPAExecution(EntityCoherentPath):
    """Same equations, computing unused controls only during case capture."""
    def forward(self, query, scores, mask, subject, obj, *, capture=False):
        if capture or self.mode == "disabled":
            return super().forward(query,scores,mask,subject,obj,capture=capture)
        c = self.config
        b,h,n,d = query.shape
        if (h,d) != (c.heads,c.head_dim) or scores.shape != (b,h,n,n):
            raise ValueError("ECPA shape mismatch")
        mask = mask.bool()
        if (~mask.any(-1)).any() or ((subject.bool()|obj.bool()) & ~mask).any() or (subject.sum(-1)==0).any() or (obj.sum(-1)==0).any():
            raise ValueError("invalid ECPA entity masks")
        z = nn.functional.normalize(query,dim=-1,eps=c.eps)
        def pool(span):
            weight = span.to(query)[:,None,:,None]
            return (z*weight).sum(-2)/weight.sum(-2)
        pair = torch.cat((pool(subject),pool(obj)),-1)
        operator = self.condition(pair).reshape(b,h,d,d)
        if self.mode == "random_phase":
            operator = self.random_operator[None].expand(b,-1,-1,-1)
        skew = .5*(operator-operator.transpose(-1,-2))
        phase = c.phase_max*torch.tanh(torch.einsum("bhid,bhde,bhje->bhij",z,skew,z))
        phase = .5*(phase-phase.transpose(-1,-2))
        position = torch.arange(n,device=query.device)
        distance = (position[:,None]-position[None,:]).abs()
        edges = (distance>0)&(distance<=2)
        support = mask[:,None,:,None]&mask[:,None,None,:]
        weights = (edges[None,None]&support).to(query)
        degree = weights.sum(-1)
        graph = weights/(degree[..., :,None]*degree[...,None,:]).clamp_min(1).sqrt()
        if self.mode == "classical":
            allowed = (edges|torch.eye(n,device=query.device,dtype=torch.bool))[None,None]&support
            step = phase.transpose(-1,-2).masked_fill(~allowed,-1e4).softmax(-1)*mask[:,None,:,None]
            probability = step
            for _ in range(c.steps-1):
                probability = probability@step
        else:
            unitary = torch.matrix_exp(-1j*c.delta*torch.polar(graph.expand(b,h,n,n),phase))
            step = unitary.abs().square() if self.mode == "dephased" else unitary
            state = step
            for _ in range(c.steps-1):
                state = step@state
            probability = (state if self.mode == "dephased" else state.abs().square()).transpose(-1,-2)
        logged = (probability+c.eps).log()
        mean = (logged*support).sum(-1,keepdim=True)/mask.sum(-1)[:,None,None,None]
        return scores+c.alpha*(logged-mean).clamp(-c.clip,c.clip)*support, {}


class ObservedQK(QuantumQueryKeyCoherentTransportKernel):
    """Original QK equations with bounded, actual two-qubit state observation."""
    def __init__(self, config):
        super().__init__(config)
        self.observe = False
        self.observations = {}

    def _single_qubit_quantum_score(self, query_angles, key_angles, *, layer_index, head_index, qubit):
        if not self.observe:
            return super()._single_qubit_quantum_score(query_angles,key_angles,
                layer_index=layer_index,head_index=head_index,qubit=qubit)
        q = torch.stack((torch.cos(query_angles/2),torch.sin(query_angles/2)),-1)
        k = torch.stack((torch.cos(key_angles/2),torch.sin(key_angles/2)),-1)
        dtype = torch.complex128 if query_angles.dtype == torch.float64 else torch.complex64
        state = torch.einsum("na,nb->nab",q,k).to(dtype)
        states = [state.reshape(-1,4)]
        for depth in range(self.config.depth):
            phase = self.config.max_phase*torch.tanh(self.raw_phase[layer_index,head_index,depth,qubit].to(query_angles))
            phase_vector = torch.stack((torch.ones_like(query_angles),torch.ones_like(query_angles),
                torch.ones_like(query_angles),torch.exp(1j*phase).expand_as(query_angles)),-1).to(dtype)
            state = (state.reshape(-1,4)*phase_vector).reshape(-1,2,2)
            states.append(state.reshape(-1,4))
            post_q = self.config.max_post_rotation*torch.tanh(self.raw_post_rotation[layer_index,head_index,depth,0,qubit].to(query_angles))
            post_k = self.config.max_post_rotation*torch.tanh(self.raw_post_rotation[layer_index,head_index,depth,1,qubit].to(query_angles))
            state = torch.einsum("nij,njk->nik",_ry_matrix(post_q).to(dtype).expand(state.shape[0],-1,-1),state)
            state = torch.einsum("nac,nbc->nab",state,_ry_matrix(post_k).to(dtype).expand(state.shape[0],-1,-1))
            states.append(state.reshape(-1,4))
        prefix = f"layer{layer_index}_qk_head{head_index}"
        self.observations[prefix+"_angles"] = torch.stack((query_angles,key_angles),-1).detach().cpu()
        self.observations[prefix+"_states"] = torch.stack(states,1).detach().cpu()
        transformed = state.reshape(-1,4)@self.transport_observable.to(state).transpose(0,1)
        return (state.reshape(-1,4).conj()*transformed).sum(-1).real


class _Observer(AttentionInterventionPassThrough):
    def __init__(self, owner, index):
        super().__init__()
        object.__setattr__(self,"owner",owner)
        self.index = index

    def after_scores(self, scores, query, key, value, *, attention_mask=None, layer_index=0):
        owner = self.owner
        if owner.scoped is None:
            raise RuntimeError("formal adapter called outside scoped forward")
        subject,obj = owner.scoped
        adjusted, extra = scores, {}
        if owner.mode == "original_qk":
            residual = owner.kernel(query,key,scores=scores,layer_index=layer_index,
                attention_mask=attention_mask,subject_mask=subject,object_mask=obj)
            adjusted = scores+residual
            if owner.capture:
                extra = dict(qk_raw=owner.kernel.last_raw_score,qk_residual=residual)
        elif owner.kernel is not None and self.index == 0:
            if owner.mode == "entity_permuted":
                subject,obj = obj,subject
            adjusted, extra = owner.kernel(query,scores,attention_mask,subject,obj,capture=owner.capture)
        if owner.capture:
            masked = adjusted.masked_fill(~attention_mask[:,None,None,:].bool(),torch.finfo(adjusted.dtype).min)
            extra.update(query=query,key=key,value=value,base_scores=scores,adjusted_scores=adjusted,
                         attention=masked.softmax(-1))
            for name,tensor in extra.items():
                owner.observed[f"layer{self.index}_{name}"] = tensor[0].detach().cpu().clone()
        return adjusted,value


class ECPAFormalModel(nn.Module):
    def __init__(self, base, mode, config, *, train_base=False):
        super().__init__()
        if base.config.dropout != 0 or base.model_parallel_enabled:
            raise ValueError("formal equivalence requires dropout=0 and one worker device")
        self.base,self.mode = base,mode
        self.kernel = None
        base.requires_grad_(train_base)
        if mode == "original_qk":
            self.kernel = ObservedQK(QueryKeyCoherentTransportConfig(
                num_layers=base.config.num_layers,num_heads=base.config.num_heads,
                head_dim=base.config.dim//base.config.num_heads,pair_chunk_size=None,seed=config["seed"]+307))
        elif mode not in ("baseline","disabled"):
            choice = "quantum" if mode == "entity_permuted" else mode
            self.kernel = ECPAExecution(ECPAConfig(heads=base.config.num_heads,
                head_dim=base.config.dim//base.config.num_heads,**config["ecpa"]),choice)
        self.scoped,self.capture,self.observed = None,False,{}
        self.recompute = False
        for index,layer in enumerate(base.encoder.layers):
            if type(layer.attn.attention_intervention) is not AttentionInterventionPassThrough:
                raise ValueError("refuse stacked attention interventions")
            layer.attn.attention_intervention = _Observer(self,index)

    def _forward(self, input_ids, attention_mask, subject_mask, object_mask, *, capture=False):
        if self.scoped is not None:
            raise RuntimeError("concurrent/reentrant formal model use")
        self.scoped,self.capture = (subject_mask,object_mask),capture
        self.observed = {}
        handles = []
        def observe(name, before=False):
            def save(module, args, output=None):
                tensor = args[0] if before else output
                self.observed[name] = tensor[0].detach().cpu().clone()
            return save
        if capture:
            encoder = self.base.encoder
            for module,name in ((encoder.token_embedding,"token_embedding"),(encoder.position_embedding,"position_embedding"),(encoder.dropout,"embedding")):
                handles.append(module.register_forward_hook(observe(name)))
            for index,layer in enumerate(encoder.layers):
                for module,name in ((layer.attn,"attention_output"),(layer.attn_norm,"attention_hidden"),(layer.ffn,"ffn"),(layer,"hidden")):
                    handles.append(module.register_forward_hook(observe(f"layer{index}_{name}")))
                handles.append(layer.attn.out_proj.register_forward_pre_hook(observe(f"layer{index}_context",True)))
            handles.append(self.base.classifier.register_forward_pre_hook(observe("pooled",True)))
            handles.append(self.base.classifier[1].register_forward_hook(observe("classifier_hidden")))
            if isinstance(self.kernel,ObservedQK):
                self.kernel.observe,self.kernel.observations = True,{}
        try:
            logits = self.base(input_ids,attention_mask,subject_mask,object_mask)
            if capture:
                self.observed.update({name:t[0].detach().cpu().clone() for name,t in dict(input_ids=input_ids,
                    attention_mask=attention_mask,subject_mask=subject_mask,object_mask=object_mask,
                    logits=logits,probabilities=logits.softmax(-1)).items()})
                if isinstance(self.kernel,ObservedQK):
                    self.observed.update(self.kernel.observations)
            return logits
        finally:
            for handle in handles:
                handle.remove()
            self.scoped,self.capture = None,False
            if isinstance(self.kernel,ObservedQK):
                self.kernel.observe = False

    def forward(self, input_ids, attention_mask, subject_mask, object_mask, *, capture=False):
        args = (input_ids,attention_mask,subject_mask,object_mask)
        if self.recompute and torch.is_grad_enabled() and not capture:
            # Scope is established again during recomputation, not left dangling
            # after the original forward. Non-reentrant mode supports frozen inputs.
            return checkpoint(self._forward,*args,use_reentrant=False)
        return self._forward(*args,capture=capture)
