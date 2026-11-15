"""Relation-hypothesis binary measurements and fixed-checkpoint diagnostics."""
import hashlib
import math
import torch
from torch import nn
from .frozen_feature_head import FrozenFeatureHead

MODES=('uniform','relation_softmax','real_overlap','quantum','dephased')
EPS=1e-6
QUBITS=3
DEPTH=2


def circuit(angles):
    """RY then RZ per wire, then directed CNOT ring, twice. Wire0 is MSB."""
    if angles.shape[-3:]!=(DEPTH,QUBITS,2):raise ValueError('Expected [...,2,3,2] angles')
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    state=torch.zeros(angles.shape[:-3]+(8,),device=angles.device,dtype=dtype)
    state[...,0]=1
    for layer in range(DEPTH):
        for wire in range(QUBITS):
            shape=state.shape
            pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=pair.unbind(-2)
            y,z=angles[...,layer,wire,:].unbind(-1)
            c,s=torch.cos(y/2),torch.sin(y/2)
            c=c[...,None,None];s=s[...,None,None]
            minus=torch.exp(-.5j*z)[...,None,None];plus=torch.exp(.5j*z)[...,None,None]
            state=torch.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),dim=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state


class RMEAHead(FrozenFeatureHead):
    """All forty effects are separate binary measurements, not one joint POVM."""
    def __init__(self,mode):
        if mode not in MODES:raise ValueError('Unknown RMEA condition')
        super().__init__(mode if mode in MODES[:3] else 'uniform')
        self.mode=mode
        if mode in MODES[3:]:
            self.prepare=nn.Linear(1028,DEPTH*QUBITS*2)
            self.effect=nn.Parameter(torch.randn(40,DEPTH,QUBITS,2)*.2)

    def forward(self,h,roles,mask,details=False):
        if self.mode in MODES[:3]:
            logits,alpha=super().forward(h,roles,mask)
            if not details:return logits,alpha
            values=self.value(h)
            return dict(logits=logits,alpha=alpha,values=values,
                        utilities=torch.einsum('btd,rd->brt',values,self.decoder))
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,):raise ValueError('Bad features')
        if mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():raise ValueError('Bad valid mask')
        angles=math.pi*torch.tanh(self.prepare(torch.cat((h,roles),-1))).reshape(h.shape[:2]+(2,3,2))
        effect_angles=math.pi*torch.tanh(self.effect)
        psi=circuit(angles);phi=circuit(effect_angles)
        if self.mode=='quantum':
            probability=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
        else:
            probability=torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
        raw=(probability+EPS)*mask[:,None,:]
        normalizer=raw.sum(-1,keepdim=True);alpha=raw/normalizer
        values=self.value(h);utilities=torch.einsum('btd,rd->brt',values,self.decoder)
        logits=(alpha*utilities).sum(-1)+self.bias
        if not details:return logits,alpha
        return dict(logits=logits,alpha=alpha,values=values,utilities=utilities,
                    angles=angles,effect_angles=effect_angles,psi=psi,phi=phi,
                    probability=probability,normalizer=normalizer,
                    sensitivity=(utilities-(alpha*utilities).sum(-1,keepdim=True))/normalizer*mask[:,None,:])


def context_permutation(mask,roles,ids):
    """One label/outcome/seed-blind permutation, shared by all relation rows."""
    if len(ids)!=len(mask):raise ValueError('Sample identities required')
    result=[]
    for i,sample_id in enumerate(ids):
        eligible=torch.where(mask[i] & (roles[i,:,2]==0) & (roles[i,:,3]==0))[0].tolist()
        order=sorted(eligible,key=lambda j:hashlib.sha256(('rmea-context-v1:'+str(sample_id)+':'+str(j)).encode()).digest())
        if len(order)>1 and order==eligible:order=order[1:]+order[:1]
        permutation=list(range(mask.shape[1]))
        for dest,src in zip(eligible,order):permutation[dest]=src
        result.append(permutation)
    return torch.tensor(result,device=mask.device,dtype=torch.long)


def replay(head,detail,mask,roles,ids):
    alpha=detail['alpha'];u=detail['utilities'];logits=detail['logits']
    permutation=context_permutation(mask,roles,ids)
    shuffled=alpha.gather(-1,permutation[:,None,:].expand_as(alpha))
    tied=alpha.mean(1,keepdim=True).expand_as(alpha)
    shuffled_logits=(shuffled*u).sum(-1)+head.bias
    tied_logits=(tied*u).sum(-1)+head.bias
    identity=logits-shuffled_logits-((alpha-shuffled)*u).sum(-1)
    return dict(shuffled_alpha=shuffled,shuffled_logits=shuffled_logits,tied_alpha=tied,
                tied_logits=tied_logits,permutation=permutation,identity_residual=identity,
                eligible_context_mask=mask & (roles[...,2]==0) & (roles[...,3]==0))
