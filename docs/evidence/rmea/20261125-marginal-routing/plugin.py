"""Two local state views route disjoint signed value coordinates."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_marginal_split','population_marginal_split','quantum_marginal_tied','real12_marginal_split')

def marginals(state,dimensions):
    a,b,c=dimensions
    if state.shape[-1]!=a*b*c:raise ValueError('State dimension mismatch')
    ab=state.reshape(state.shape[:-1]+(a*b,c))
    bc=state.reshape(state.shape[:-1]+(a,b*c))
    return (ab@ab.conj().transpose(-1,-2),bc.transpose(-1,-2)@bc.conj())

def routing(states,queries,dimensions,mask,population=False):
    maps=[]
    for rho,sigma in zip(marginals(states,dimensions),marginals(queries,dimensions)):
        if population:
            score=torch.einsum('ri,bti->brt',sigma.diagonal(dim1=-2,dim2=-1).real,rho.diagonal(dim1=-2,dim2=-1).real)
        else:score=torch.einsum('rij,btji->brt',sigma,rho).real
        raw=(score.clamp_min(0)+EPS)*mask[:,None]
        maps.append(raw/raw.sum(-1,keepdim=True))
    return torch.stack(maps,-1)

def readout(alpha,utility,bias,tied=False):
    if tied:alpha=alpha.mean(-1,keepdim=True).expand_as(alpha)
    return (alpha*utility).sum((-1,-2))+bias

class MarginalRouting(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head

    def components(self,h,roles,mask):
        if h.ndim!=3 or h.shape[0]==0 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():
            raise ValueError('Invalid word inputs')
        a=self.head
        if self.kind=='real12_marginal_split':
            states=F.normalize(a.key(torch.cat((h,roles),-1)),dim=-1)
            queries=F.normalize(a.query,dim=-1);dims=(2,3,2)
        else:
            angles=(math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2))
            states=circuit(angles);queries=circuit(math.pi*torch.tanh(a.effect));dims=(2,2,2)
        alpha=routing(states,queries,dims,mask,self.kind=='population_marginal_split')
        v=a.value(h)
        utility=torch.stack([torch.einsum('btd,rd->brt',v[...,k::2],a.decoder[:,k::2]) for k in range(2)],-1)
        return states,queries,dims,alpha,utility

    def forward(self,h,roles,mask):
        _,_,_,alpha,utility=self.components(h,roles,mask)
        return readout(alpha,utility,self.head.bias,self.kind=='quantum_marginal_tied')

def make_head(kind):return MarginalRouting(kind,Head('real12' if kind=='real12_marginal_split' else 'quantum'))

def initialize(initial,seed):
    return {k:MarginalRouting(k,initialize_head('real12' if k=='real12_marginal_split' else 'quantum',initial,dict(seed=seed,training_dtype='float32'))) for k in MODES}
