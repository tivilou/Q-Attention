"""Scientific plugin only: conditional channel, unchanged native task readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_midgate','quantum_outgate','quantum_midfixed','classical_gate')


def layer(angles,state):
    for wire in range(3):
        shape=state.shape
        pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
        a,b=pair.unbind(-2);y,z=angles[...,wire,:].unbind(-1)
        c,s=torch.cos(y/2)[...,None,None],torch.sin(y/2)[...,None,None]
        minus=torch.exp(-.5j*z)[...,None,None];plus=torch.exp(.5j*z)[...,None,None]
        state=torch.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),dim=-2).reshape(shape)
    for control,target in ((0,1),(1,2),(2,0)):
        perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
        state=state[...,perm]
    return state


def layer_matrix(angles):
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    identity=torch.eye(8,device=angles.device,dtype=dtype).expand(angles.shape[:-2]+(8,8))
    return layer(angles[...,None,:,:],identity).transpose(-1,-2)


class ChannelHead(nn.Module):
    def __init__(self,kind,native):
        super().__init__();assert kind in MODES;self.kind=kind;self.head=native.head
        if kind!='quantum_midfixed':
            self.gate=nn.Linear(12,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def components(self,h,roles,mask):
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,) or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():raise ValueError('Invalid word inputs')
        a=self.head
        if self.kind=='classical_gate':
            features=a.key(torch.cat((h,roles),-1));k=F.normalize(features,dim=-1);q=F.normalize(a.query,dim=-1)
            coherent=torch.einsum('btd,rd->brt',k,q).square()
            endpoint=torch.einsum('btd,rd->brt',k.square(),q.square())
            features=torch.tanh(features)
        else:
            features=torch.tanh(a.prepare(torch.cat((h,roles),-1)))
            theta=(math.pi*features).reshape(h.shape[:2]+(2,3,2))
            dtype=torch.complex128 if h.dtype==torch.float64 else torch.complex64
            zero=torch.zeros(h.shape[:2]+(8,),dtype=dtype,device=h.device);zero[...,0]=1
            psi1=layer(theta[...,0,:,:],zero);psi=layer(theta[...,1,:,:],psi1)
            phi=circuit(math.pi*torch.tanh(a.effect))
            coherent=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
            if self.kind=='quantum_outgate':
                endpoint=torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
            else:
                u2=layer_matrix(theta[...,1,:,:])
                pulled=torch.einsum('btoi,ro->brti',u2.conj(),phi)
                endpoint=torch.einsum('brti,bti->brt',pulled.abs().square(),psi1.abs().square())
        lam=torch.ones(h.shape[:2],dtype=h.dtype,device=h.device) if self.kind=='quantum_midfixed' else self.gate(features).sigmoid().squeeze(-1)
        return coherent,endpoint,lam

    def forward(self,h,roles,mask):
        coherent,endpoint,lam=self.components(h,roles,mask)
        probability=(1-lam[:,None])*coherent+lam[:,None]*endpoint
        raw=(probability+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        u=torch.einsum('btd,rd->brt',self.head.value(h),self.head.decoder)
        return (alpha*u).sum(-1)+self.head.bias


def make_head(kind):return ChannelHead(kind,Head('real12' if kind=='classical_gate' else 'quantum'))


def initialize(initial,seed):
    models={}
    for kind in MODES:
        base=initialize_head('real12' if kind=='classical_gate' else 'quantum',initial,dict(seed=seed,training_dtype='float32'))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);models[kind]=ChannelHead(kind,base)
    return models
