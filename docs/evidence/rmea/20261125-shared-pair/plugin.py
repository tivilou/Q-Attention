"""Shared positive comparison attention for signed relation-decoder differences."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_pair','quantum_pair_intensity','real12_pair')


def pair_weights(amplitudes,queries,first,second,intensity=False):
    qr,qs=queries[first],queries[second]
    c=(qr.conj()*qs).sum(-1);radius=c.abs();denom=radius.clamp_min(EPS)
    eta=c/denom;t=radius/denom
    ar,ass=amplitudes[:,first],amplitudes[:,second]
    if intensity:
        # Exact trace of the positive implemented effect, robust at coincident rays.
        trace=((qr-eta.conj()[:,None]*qs).abs().square().sum(-1)
               +(1-t.square())*qs.abs().square().sum(-1))/2
        base_trace=(qr.abs().square().sum(-1)+qs.abs().square().sum(-1))/2
        return ((ar.abs().square()+ass.abs().square())/2)*(trace/base_trace)[None,:,None]
    return ((ar-eta[None,:,None]*ass).abs().square()
            +(1-t.square())[None,:,None]*ass.abs().square())/2


def pool_pairs(amplitudes,queries,utilities,mask,bias,intensity=False,block=128):
    classes=queries.shape[0];first,second=torch.triu_indices(classes,classes,offset=1,device=queries.device)
    out=torch.zeros((amplitudes.shape[0],classes),dtype=utilities.dtype,device=utilities.device)
    for start in range(0,len(first),block):
        r,s=first[start:start+block],second[start:start+block]
        w=pair_weights(amplitudes,queries,r,s,intensity)
        raw=(w+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        edge=(alpha*(utilities[:,r]-utilities[:,s])).sum(-1)
        # Dense incidence multiplication avoids nondeterministic CUDA atomic votes.
        incidence=F.one_hot(r,classes).to(utilities.dtype)-F.one_hot(s,classes).to(utilities.dtype)
        out=out+edge@incidence
    return out/classes+bias


class PairHead(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head

    def components(self,h,roles,mask):
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():raise ValueError('Invalid word inputs')
        a=self.head
        if self.kind=='real12_pair':
            state=F.normalize(a.key(torch.cat((h,roles),-1)),dim=-1);query=F.normalize(a.query,dim=-1)
        else:
            theta=math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1))).reshape(h.shape[:2]+(2,3,2))
            state=circuit(theta);query=circuit(math.pi*torch.tanh(a.effect))
        amplitude=torch.einsum('rd,btd->brt',query.conj(),state)
        utility=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
        return amplitude,query,utility

    def forward(self,h,roles,mask):
        amplitude,query,utility=self.components(h,roles,mask)
        return pool_pairs(amplitude,query,utility,mask,self.head.bias,self.kind=='quantum_pair_intensity')


def make_head(kind):return PairHead(kind,Head('real12' if kind=='real12_pair' else 'quantum'))


def initialize(initial,seed):
    return {k:PairHead(k,initialize_head('real12' if k=='real12_pair' else 'quantum',initial,
            dict(seed=seed,training_dtype='float32'))) for k in MODES}
