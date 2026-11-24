"""Shared entity-conditioned unitary on the existing relation-query frame."""
import math
import torch
from .rmea import circuit
from .rmea_entity_query import anchor,real_circuit
from .rmea_matched_routers import Head,initialize_head as native_initialize

MODES=('quantum_transport','dephased_transport','real_circuit_transport')
def native_mode(kind):
    if kind not in MODES:raise ValueError(kind)
    return 'dephased' if kind=='dephased_transport' else 'quantum'

def transport(angles,state,*,real=False):
    """Apply the native complex circuit, or 4-layer real control, to an input state.

    angles[...,12], state[...,8]; their leading axes must already agree.
    This is a unitary circuit on amplitudes, not addition before query tanh.
    """
    if angles.shape[:-1]!=state.shape[:-1] or angles.shape[-1]!=12 or state.shape[-1]!=8:
        raise ValueError('Incompatible transport axes')
    depth=4 if real else 2
    for layer in range(depth):
        for wire in range(3):
            shape=state.shape;pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=pair.unbind(-2)
            y=angles[...,layer*3+wire] if real else angles[...,layer*6+wire*2]
            c=y.div(2).cos()[...,None,None];s=y.div(2).sin()[...,None,None]
            first,second=c*a-s*b,s*a+c*b
            if not real:
                z=angles[...,layer*6+wire*2+1]
                first=first*torch.exp(-.5j*z)[...,None,None]
                second=second*torch.exp(.5j*z)[...,None,None]
            state=torch.stack((first,second),-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state

class QueryTransportHead(Head):
    def __init__(self,kind):
        super().__init__(native_mode(kind));self.kind=kind

    def states(self,h,roles,mask,*,conditioned=True):
        if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
            or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word input')
        a=self.head;real=self.kind=='real_circuit_transport';x=torch.cat((h,roles),-1)
        ka=math.pi*a.prepare(x).tanh();qa=math.pi*a.effect.flatten(-3).tanh()
        key=real_circuit(ka) if real else circuit(ka.reshape(h.shape[:2]+(2,3,2)))
        query=real_circuit(qa) if real else circuit(qa.reshape(40,2,3,2))
        query=query[None].expand(len(h),-1,-1)
        if conditioned:
            ca=math.pi*a.prepare(anchor(x,roles,mask)).tanh()
            query=transport(ca[:,None,:].expand(-1,40,-1),query,real=real)
        return key,query

    def forward(self,h,roles,mask,*,conditioned=True):
        key,query=self.states(h,roles,mask,conditioned=conditioned);a=self.head
        if self.mode=='dephased':score=torch.einsum('brd,btd->brt',query.abs().square(),key.abs().square())
        else:score=torch.einsum('brd,btd->brt',query.conj(),key).abs().square()
        raw=(score+1e-6)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize_head(kind,initial,cfg):
    native=native_initialize(native_mode(kind),initial,cfg)
    with torch.random.fork_rng(devices=[]):model=QueryTransportHead(kind).to(dtype=getattr(torch,cfg['training_dtype']))
    model.load_state_dict(native.state_dict());return model

def validate_initialized_head(model,kind,initial,cfg):
    from .rmea_matched_routers import validate_initialized_head as validate
    mode=native_mode(kind);mapped=dict(cfg);mapped['parameter_schemas']={mode:cfg['parameter_schemas'][kind]}
    validate(model,mode,initial,mapped)
