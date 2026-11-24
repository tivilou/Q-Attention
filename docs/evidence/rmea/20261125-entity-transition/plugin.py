"""Entity-conditioned full-unitary transition attention; no runtime changes."""
import math
import torch
from torch import nn
from baselines.retacred.rmea import circuit, EPS
from baselines.retacred.rmea_matched_routers import Head, initialize_head
from circuit_ops import quantum_matrix, orthogonal_transition

MODES=('quantum_transition','population_transition','quantum_drop_subject','real12_transition')

def mixture(states, selection):
    if selection.dtype!=torch.bool or selection.shape!=states.shape[:2] or not selection.any(-1).all():
        raise ValueError('Missing or invalid entity span')
    w=selection.to(states.dtype)/selection.sum(-1,keepdim=True)
    return torch.einsum('bt,bti,btj->bij',w,states,states.conj())

def probabilities(transition, relation, subject, obj, population=False):
    if population:
        out=torch.einsum('btij,bj->bti',transition.abs().square(),subject.diagonal(dim1=-2,dim2=-1).real)
        effect=torch.einsum('rij,bj->bri',relation.abs().square(),obj.diagonal(dim1=-2,dim2=-1).real)
        return torch.einsum('bri,bti->brt',effect,out)
    out=transition@subject[:,None]@transition.conj().transpose(-1,-2)
    effect=relation[None]@obj[:,None]@relation.conj().transpose(-1,-2)[None]
    return torch.einsum('brij,btji->brt',effect,out).real

def pool(p,mask,utility,bias):
    raw=(p.clamp_min(0)+EPS)*mask[:,None]
    alpha=raw/raw.sum(-1,keepdim=True)
    return (alpha*utility).sum(-1)+bias

class EntityTransition(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head

    def components(self,h,roles,mask):
        if h.ndim!=3 or h.shape[0]==0 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():
            raise ValueError('Invalid word inputs')
        for k in (2,3):
            f=roles[...,k][mask]
            if not ((f==0)|(f==1)).all():raise ValueError('Invalid entity flag')
        a=self.head
        if self.kind=='real12_transition':
            angle=math.pi*torch.tanh(a.key(torch.cat((h,roles),-1)))
            unitary=orthogonal_transition(angle)
            relation=orthogonal_transition(math.pi*torch.tanh(a.query))
        else:
            angle=(math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2))
            unitary=quantum_matrix(angle)
            zero=quantum_matrix(angle.new_zeros(2,3,2))
            relation=quantum_matrix(math.pi*torch.tanh(a.effect))@zero.conj().T
        states=unitary[...,0]
        transition=unitary if self.kind=='real12_transition' else unitary@zero.conj().T
        subject=mixture(states,mask&(roles[...,2]==1))
        obj=mixture(states,mask&(roles[...,3]==1))
        if self.kind=='quantum_drop_subject':
            subject=torch.zeros_like(subject);subject[:,0,0]=1
        utility=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
        return states,transition,relation,subject,obj,utility

    def forward(self,h,roles,mask):
        _,transition,relation,subject,obj,utility=self.components(h,roles,mask)
        return pool(probabilities(transition,relation,subject,obj,self.kind=='population_transition'),mask,utility,self.head.bias)

def make_head(kind):return EntityTransition(kind,Head('real12' if kind=='real12_transition' else 'quantum'))

def initialize(initial,seed):
    return {k:EntityTransition(k,initialize_head('real12' if k=='real12_transition' else 'quantum',initial,dict(seed=seed,training_dtype='float32'))) for k in MODES}
