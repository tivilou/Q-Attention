"""Fixed12-angle register allocation and ordinary classical kernel controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_native','quantum_register6','dephased_register6','real_product6','rbf12','real12','softmax12')
BASE={m:('quantum' if m in MODES[:3] else 'real12' if m in ('real_product6','rbf12') else m) for m in MODES}

def qubit_state(angles):
    y,z=angles.unbind(-1)
    return torch.stack((torch.cos(y/2)*torch.exp(-.5j*z),torch.sin(y/2)*torch.exp(.5j*z)),-1)

def product_probability(key,query,dephased=False):
    if dephased:
        ky=key[...,0];qy=query[...,0]
        k=torch.stack((torch.cos(ky/2).square(),torch.sin(ky/2).square()),-1)
        q=torch.stack((torch.cos(qy/2).square(),torch.sin(qy/2).square()),-1)
        factors=torch.einsum('rkd,btkd->brtk',q,k)
    else:
        k=qubit_state(key);q=qubit_state(query)
        factors=torch.einsum('rkd,btkd->brtk',q.conj(),k).abs().square()
    return factors.prod(-1)

class RegisterHead(nn.Module):
    def __init__(self,mode,native=None):
        super().__init__();assert mode in MODES;self.mode=mode
        self.native=native if native is not None else Head(BASE[mode])
    @property
    def head(self):return self.native.head
    def route_alpha(self,h,roles,mask):
        if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
                or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid shared frozen-word input')
        a=self.head;x=torch.cat((h,roles),-1)
        if self.mode in ('quantum_register6','dephased_register6'):
            k=(math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(6,2))
            q=(math.pi*torch.tanh(a.effect)).reshape(40,6,2)
            p=product_probability(k,q,self.mode=='dephased_register6')
        elif self.mode=='real_product6':
            k=F.normalize(a.key(x).reshape(h.shape[:2]+(6,2)),dim=-1)
            q=F.normalize(a.query.reshape(40,6,2),dim=-1)
            p=torch.einsum('rkd,btkd->brtk',q,k).square().prod(-1)
        else:
            score=-.5*(a.key(x)[:,None]-a.query[None,:,None]).square().mean(-1)
            alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        if self.mode!='rbf12':
            raw=(p+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        return alpha
    def forward(self,h,roles,mask):
        if self.mode in ('quantum_native','real12','softmax12'):return self.native(h,roles,mask)
        a=self.head;alpha=self.route_alpha(h,roles,mask)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
    return {m:RegisterHead(m,initialize_head(BASE[m],initial,dict(seed=seed,training_dtype='float32'))) for m in MODES}
