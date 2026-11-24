"""Input-computable token priors; native measurement and shared value unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_prior','dephased_prior','real12_prior','softmax12_prior','rbf12_no_energy','quantum_native','real12','softmax12','rbf12')
BASE=dict(zip(MODES,('quantum','dephased','real12','softmax12','real12','quantum','real12','softmax12','real12')))

class PriorHead(nn.Module):
    def __init__(self,mode,native=None):
        super().__init__();assert mode in MODES;self.mode=mode
        self.native=native if native is not None else Head(BASE[mode])
    @property
    def head(self):return self.native.head
    def routing(self,h,roles,mask):
        if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
                or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid shared frozen-word input')
        a=self.head;x=torch.cat((h,roles),-1)
        if self.mode=='rbf12':
            score=-.5*(a.key(x)[:,None]-a.query[None,:,None]).square().mean(-1)
            return score.masked_fill(~mask[:,None],-torch.inf).softmax(-1),None,None,None
        if self.mode in ('quantum_prior','dephased_prior'):
            detail=a(h,roles,mask,details=True)
            key=a.prepare(x);log_likelihood=torch.log(detail['probability']+EPS)
        else:
            key=a.key(x)
            if self.mode=='real12_prior':
                p=torch.einsum('btd,rd->brt',F.normalize(key,dim=-1),F.normalize(a.query,dim=-1)).square()
                log_likelihood=torch.log(p+EPS)
            else:
                scale=12 if self.mode=='rbf12_no_energy' else math.sqrt(12)
                log_likelihood=torch.einsum('btd,rd->brt',key,a.query)/scale
        energy=key.square().mean(-1)/2
        score=log_likelihood if self.mode=='rbf12_no_energy' else log_likelihood-energy[:,None]
        return score.masked_fill(~mask[:,None],-torch.inf).softmax(-1),log_likelihood,energy,(detail if self.mode in ('quantum_prior','dephased_prior') else None)
    def forward(self,h,roles,mask):
        if self.mode in ('quantum_native','real12','softmax12'):return self.native(h,roles,mask)
        alpha,_,_,detail=self.routing(h,roles,mask);a=self.head
        if detail is not None:return (alpha*detail['utilities']).sum(-1)+a.bias
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
    return {m:PriorHead(m,initialize_head(BASE[m],initial,dict(seed=seed,training_dtype='float32'))) for m in MODES}
