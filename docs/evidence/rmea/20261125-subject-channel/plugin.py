"""Relation-conditioned unitary transport of an input subject density."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head
from circuit_ops import quantum_matrix,orthogonal_transition

MODES=('quantum_subject_channel','population_subject_channel','real12_subject_channel')

def source_density(states,source_mask):
    if source_mask.shape!=states.shape[:2] or source_mask.dtype!=torch.bool or not source_mask.any(-1).all():
        raise ValueError('Missing or invalid subject positions')
    weight=source_mask.to(states.dtype)/source_mask.sum(-1,keepdim=True)
    return torch.einsum('bt,bti,btj->bij',weight,states,states.conj())

def channel_effect(unitary,density):
    return torch.einsum('rik,bkl,rjl->brij',unitary,density,unitary.conj())

def channel_scores(states,effect,population=False):
    if population:
        return torch.einsum('bri,bti->brt',effect.diagonal(dim1=-2,dim2=-1).real,states.abs().square()).clamp_min(0)
    return torch.einsum('bti,brij,btj->brt',states.conj(),effect,states).real.clamp_min(0)

def pool_scores(probability,mask,utilities,bias):
    raw=(probability+EPS)*mask[:,None]
    alpha=raw/raw.sum(-1,keepdim=True)
    return (alpha*utilities).sum(-1)+bias

class SubjectChannel(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head

    def components(self,h,roles,mask):
        if h.ndim!=3 or h.shape[0]==0 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():
            raise ValueError('Invalid word inputs')
        flags=roles[...,2]
        if not ((flags[mask]==0)|(flags[mask]==1)).all():raise ValueError('Invalid subject flag')
        source=mask&(flags==1);a=self.head
        if self.kind=='real12_subject_channel':
            states=F.normalize(a.key(torch.cat((h,roles),-1)),dim=-1)
            unitary=orthogonal_transition(math.pi*torch.tanh(a.query))
        else:
            angles=(math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2))
            states=circuit(angles);unitary=quantum_matrix(math.pi*torch.tanh(a.effect))
        density=source_density(states,source)
        effect=channel_effect(unitary,density)
        utility=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
        return states,unitary,density,effect,utility

    def forward(self,h,roles,mask):
        states,_,_,effect,utility=self.components(h,roles,mask)
        p=channel_scores(states,effect,self.kind=='population_subject_channel')
        return pool_scores(p,mask,utility,self.head.bias)

def make_head(kind):return SubjectChannel(kind,Head('real12' if kind=='real12_subject_channel' else 'quantum'))

def initialize(initial,seed):
    return {k:SubjectChannel(k,initialize_head('real12' if k=='real12_subject_channel' else 'quantum',initial,dict(seed=seed,training_dtype='float32'))) for k in MODES}
