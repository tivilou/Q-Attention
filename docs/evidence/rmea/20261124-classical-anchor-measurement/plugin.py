"""Native real route plus a centered independently initialized route correction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import RMEAHead,circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('real_native','soft_native','real_plus_quantum','real_plus_dephased','real_plus_real12','real_plus_softmax12')
BRANCH={m:m.removeprefix('real_plus_') for m in MODES[2:]}

def branch_module(mode,seed=1013):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);a=Head(mode).head
    out=nn.Module()
    if mode in ('quantum','dephased'):
        out.prepare=a.prepare;out.effect=a.effect
    else:out.key=a.key;out.query=a.query
    return out

def attention(a,mode,h,roles,mask):
    x=torch.cat((h,roles),-1)
    if mode in ('quantum','dephased'):
        key=circuit((math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(2,3,2)))
        query=circuit(math.pi*torch.tanh(a.effect))
        p=(torch.einsum('rd,btd->brt',query.conj(),key).abs().square() if mode=='quantum' else
           torch.einsum('rd,btd->brt',query.abs().square(),key.abs().square()))
    elif mode=='softmax12':
        score=torch.einsum('btd,rd->brt',a.key(x),a.query)/math.sqrt(12)
        return score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
    else:
        p=torch.einsum('btd,rd->brt',F.normalize(a.key(x),dim=-1),F.normalize(a.query,dim=-1)).square()
    raw=(p+EPS)*mask[:,None];return raw/raw.sum(-1,keepdim=True)

class AnchoredMeasurement(nn.Module):
    def __init__(self,mode,native=None):
        super().__init__();assert mode in MODES;self.mode=mode
        self.native=native if native is not None else Head('softmax12' if mode=='soft_native' else 'real12')
        if mode in BRANCH:
            self.route=branch_module(BRANCH[mode]);self.gate=nn.Parameter(torch.zeros(()))
    @property
    def head(self):return self.native.head
    def forward(self,h,roles,mask):
        base=self.native(h,roles,mask)
        if self.mode not in BRANCH:return base
        alpha=attention(self.route,BRANCH[self.mode],h,roles,mask)
        uniform=mask[:,None].to(h.dtype)/mask.sum(-1)[:,None,None]
        u=torch.einsum('btd,rd->brt',self.head.value(h),self.head.decoder)
        return base+self.gate*((alpha-uniform)*u).sum(-1)

def initialize(initial,seed):
    return {m:AnchoredMeasurement(m,initialize_head('softmax12' if m=='soft_native' else 'real12',initial,
            dict(seed=seed,training_dtype='float32'))) for m in MODES}
