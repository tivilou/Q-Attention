"""Ordered finite memory before unchanged RMEA relation readout; no runtime code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import initialize_head

MODES=('quantum_native','quantum_seq','dephased_seq','real12_seq','softmax12_seq')
BASE=dict(zip(MODES,('quantum','quantum','dephased','real12','softmax12')))

def quantum_matrix(angles):
    """Apply the exact native circuit to all computational basis columns."""
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

def quantum_transition(angles):
    # U(0) is a permutation. U(theta)U(0)^H permutes columns by this row map.
    rowmap=list(range(8))
    for _ in range(2):
        for control,target in ((0,1),(1,2),(2,0)):
            p=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            rowmap=[rowmap[i] for i in p]
    return quantum_matrix(angles)[...,rowmap]

def orthogonal_transition(angles):
    state=torch.eye(12,dtype=angles.dtype,device=angles.device).expand(angles.shape[:-1]+(12,12))
    for layer in range(2):
        order=list(range(12)) if layer==0 else list(range(1,12))+[0]
        inverse=[order.index(i) for i in range(12)]
        pair=state[...,order,:].reshape(angles.shape[:-1]+(6,2,12))
        a,b=pair.unbind(-2);theta=angles[...,6*layer:6*(layer+1)]
        c,s=theta.cos()[...,None],theta.sin()[...,None]
        state=torch.stack((c*a-s*b,s*a+c*b),dim=-2).reshape(angles.shape[:-1]+(12,12))[...,inverse,:]
    return state

def prefix_states(transitions,mask,*,serial=False):
    """Inclusive chronological product V_t ... V_0; masked steps are identity."""
    d=transitions.shape[-1]
    eye=torch.eye(d,dtype=transitions.dtype,device=transitions.device)
    matrices=torch.where(mask[...,None,None],transitions,eye)
    if serial:
        p=eye.expand(matrices.shape[0],d,d);rows=[]
        for t in range(matrices.shape[1]):
            p=matrices[:,t]@p;rows.append(p[...,0])
        return torch.stack(rows,dim=1)
    distance=1
    while distance<matrices.shape[1]:
        matrices=torch.cat((matrices[:,:distance],matrices[:,distance:]@matrices[:,:-distance]),dim=1)
        distance*=2
    return matrices[...,0]

class OrderedMemory(nn.Module):
    def __init__(self,mode,native):
        super().__init__();self.mode=mode;self.base=BASE[mode];self.native=native

    @property
    def head(self):return self.native.head

    def forward(self,h,roles,mask):
        if self.mode=='quantum_native':return self.native(h,roles,mask)
        a=self.head;x=torch.cat((h,roles),-1)
        scale=mask.sum(-1).to(h.dtype).sqrt()[:,None,None]
        if self.base in ('quantum','dephased'):
            angles=(math.pi*torch.tanh(a.prepare(x))/scale).reshape(h.shape[:2]+(2,3,2))
            matrices=quantum_transition(angles)
            if self.base=='dephased':
                key=prefix_states(matrices.abs().square(),mask)
                query=circuit(math.pi*torch.tanh(a.effect)).abs().square()
                probability=torch.einsum('rd,btd->brt',query,key)
            else:
                key=prefix_states(matrices,mask)
                query=circuit(math.pi*torch.tanh(a.effect))
                probability=torch.einsum('rd,btd->brt',query.conj(),key).abs().square()
        else:
            angles=math.pi*torch.tanh(a.key(x))/scale
            key=prefix_states(orthogonal_transition(angles),mask)
            if self.base=='softmax12':
                score=torch.einsum('rd,btd->brt',a.query,key)/math.sqrt(12)
                alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            else:
                probability=torch.einsum('rd,btd->brt',F.normalize(a.query,dim=-1),key).square()
        if self.base!='softmax12':
            raw=(probability+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
    return {mode:OrderedMemory(mode,initialize_head(BASE[mode],initial,dict(seed=seed,training_dtype='float32'))) for mode in MODES}
