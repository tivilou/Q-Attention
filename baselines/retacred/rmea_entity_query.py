"""Parameter-tied entity-pair queries; no training, storage or scheduling code."""
import math
import torch
from torch.nn import functional as F
from .rmea import circuit
from .rmea_matched_routers import Head, initialize_head as native_initialize

MODES=('quantum_context','dephased_context','real12_context','softmax12_context','real_circuit_context')


def native_mode(kind):
    if kind not in MODES:raise ValueError(kind)
    return {'quantum_context':'quantum','dephased_context':'dephased',
            'real_circuit_context':'quantum','real12_context':'real12',
            'softmax12_context':'softmax12'}[kind]


def anchor(x,roles,mask):
    parts=[]
    for j in (2,3):
        flag=roles[...,j]
        if not ((flag==0)|(flag==1)).all():raise ValueError('Nonbinary entity flag')
        selected=mask & flag.bool()
        if not selected.any(-1).all():raise ValueError('Missing subject/object word')
        weight=selected.to(x.dtype)
        parts.append((x*weight[...,None]).sum(1)/weight.sum(1)[:,None])
    return (parts[0]+parts[1])*.5


def real_circuit(angles):
    """Four RY/CNOT-ring layers, 12 real angles; wire0 is MSB."""
    state=torch.zeros(angles.shape[:-1]+(8,),dtype=angles.dtype,device=angles.device)
    state[...,0]=1
    for layer in range(4):
        for wire in range(3):
            shape=state.shape
            pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            x,y=pair.unbind(-2);a=angles[...,layer*3+wire]
            c=a.div(2).cos()[...,None,None];s=a.div(2).sin()[...,None,None]
            state=torch.stack((c*x-s*y,s*x+c*y),-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,perm]
    return state


class EntityQueryHead(Head):
    def __init__(self,kind):
        super().__init__(native_mode(kind));self.kind=kind

    def forward(self,h,roles,mask,*,conditioned=True):
        if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
            or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word input')
        a=self.head;x=torch.cat((h,roles),-1)
        c=anchor(x,roles,mask) if conditioned else None
        if self.mode in ('quantum','dephased'):
            ka=math.pi*torch.tanh(a.prepare(x))
            pre=a.effect.flatten(-3)[None].expand(len(h),-1,-1)
            if c is not None:pre=pre+a.prepare(c)[:,None,:]
            qa=math.pi*torch.tanh(pre)
            if self.kind=='real_circuit_context':
                key=real_circuit(ka);query=real_circuit(qa)
            else:
                key=circuit(ka.reshape(h.shape[:2]+(2,3,2)))
                query=circuit(qa.reshape((len(h),40,2,3,2)))
            if self.mode=='dephased':
                score=torch.einsum('brd,btd->brt',query.abs().square(),key.abs().square())
            else:score=torch.einsum('brd,btd->brt',query.conj(),key).abs().square()
        else:
            key=a.key(x);query=a.query[None].expand(len(h),-1,-1)
            if c is not None:query=query+a.key(c)[:,None,:]
            if self.mode=='softmax12':
                score=torch.einsum('brd,btd->brt',query,key)/math.sqrt(12)
            else:
                score=torch.einsum('brd,btd->brt',F.normalize(query,dim=-1),F.normalize(key,dim=-1)).square()
        if self.mode=='softmax12':alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        else:
            raw=(score+1e-6)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias


def initialize_head(kind,initial,cfg):
    native=native_initialize(native_mode(kind),initial,cfg)
    with torch.random.fork_rng(devices=[]):
        model=EntityQueryHead(kind).to(dtype=getattr(torch,cfg['training_dtype']))
    model.load_state_dict(native.state_dict())
    return model


def validate_initialized_head(model,kind,initial,cfg):
    from .rmea_matched_routers import validate_initialized_head as validate
    mode=native_mode(kind);mapped=dict(cfg)
    mapped['parameter_schemas']={mode:cfg['parameter_schemas'][kind]}
    validate(model,mode,initial,mapped)
