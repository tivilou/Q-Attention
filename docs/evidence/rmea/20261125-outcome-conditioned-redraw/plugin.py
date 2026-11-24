"""Utility-coupled decision after an eight-outcome measurement; no runtime code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import initialize_head

MODES=('quantum_record','dephased_record','utility_oracle','classical_record')

def quantum_matrix(angles):
    # Reuse the accepted ordered-memory exact native U construction, unchanged.
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    state=torch.eye(8,dtype=dtype,device=angles.device).expand(angles.shape[:-3]+(8,8))
    for layer in range(2):
        for wire in range(3):
            shape=state.shape
            pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=pair.unbind(-2)
            y,z=angles[...,layer,wire,:].unbind(-1)
            c,s=torch.cos(y/2)[...,None,None,None],torch.sin(y/2)[...,None,None,None]
            minus=torch.exp(-.5j*z)[...,None,None,None]
            plus=torch.exp(.5j*z)[...,None,None,None]
            state=torch.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),dim=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state.transpose(-1,-2)

def decision(a,p,u):
    mu=(a*u).sum(-1)
    c=(a[...,None]*p*(u-mu[...,None])[...,None]).sum(-2)
    return mu+F.relu(c).sum(-1),c

class Redraw(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head
        if kind=='classical_record':
            self.record_weight=nn.Parameter(torch.empty(40,8,13));self.record_bias=nn.Parameter(torch.zeros(40,8))
            nn.init.uniform_(self.record_weight,-1/math.sqrt(13),1/math.sqrt(13))
    def routing(self,h,roles,mask):
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():raise ValueError('Invalid frozen word inputs')
        a=self.head;u=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
        if self.kind=='classical_record':
            key=a.key(torch.cat((h,roles),-1))
            raw=(torch.einsum('btd,rd->brt',F.normalize(key,dim=-1),F.normalize(a.query,dim=-1)).square()+EPS)*mask[:,None]
            alpha=raw/raw.sum(-1,keepdim=True);mu=(alpha*u).sum(-1)
            z=torch.einsum('btd,rkd->brtk',key,self.record_weight[...,:12])
            z=z+(u-mu[...,None])[...,None]*self.record_weight[...,12][None,:,None]+self.record_bias[None,:,None]
            p=z.softmax(-1)
        else:
            angles=math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1))).reshape(h.shape[:2]+(2,3,2))
            effects=math.pi*torch.tanh(a.effect);psi=circuit(angles);phi=circuit(effects)
            if self.kind=='dephased_record':probability=torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
            else:probability=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
            raw=(probability+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
            if self.kind=='utility_oracle':return alpha,None,u
            matrix=quantum_matrix(effects)
            if self.kind=='dephased_record':p=torch.einsum('rdk,btd->brtk',matrix.abs().square(),psi.abs().square())
            else:p=torch.einsum('rdk,btd->brtk',matrix.conj(),psi).abs().square()
            # U is unitary analytically; normalize only floating-point roundoff.
            p=p/p.sum(-1,keepdim=True)
        return alpha,p,u
    def forward(self,h,roles,mask):
        a,p,u=self.routing(h,roles,mask);mu=(a*u).sum(-1)
        if self.kind=='utility_oracle':z=mu+(a*F.relu(u-mu[...,None])).sum(-1)
        else:z,_=decision(a,p,u)
        return z+self.head.bias

def initialize(initial,seed):
    result={};cfg=dict(seed=seed,training_dtype='float32')
    for kind in MODES:
        base=initialize_head('real12' if kind=='classical_record' else 'quantum',initial,cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);m=Redraw(kind,base)
        result[kind]=m
    return result
