"""Exact displaced-Fock and parity-twirled kernels; no execution or storage code."""
import torch
from .rmea_matched_routers import Head,initialize_head as native_initialize,validate_initialized_head as native_validate

MODES=('fock1','parity_fock1','vacuum_born','gaussian3_born','rbf12')

def one_photon(distance):
    return ((1-distance)*torch.exp(-distance/2)).square()

def optical_probability(key,query,kind):
    # Six complex displacements use the twelve original real coordinates.
    k=key.reshape(key.shape[:2]+(6,2));q=query.reshape((40,6,2))
    minus=(k[:,None]-q[None,:,None]).square().sum(-1)/24
    if kind=='fock1':return one_photon(minus).prod(-1)
    if kind=='parity_fock1':
        plus=(k[:,None]+q[None,:,None]).square().sum(-1)/24
        return ((one_photon(minus)+one_photon(plus))/2).prod(-1)
    if kind=='vacuum_born':return torch.exp(-minus.sum(-1))
    if kind=='gaussian3_born':return torch.exp(-3*minus.sum(-1))
    raise ValueError(kind)

class OpticalHead(Head):
    def __init__(self,kind):
        if kind not in MODES:raise ValueError(kind)
        super().__init__('real12');self.kind=kind
    def forward(self,h,roles,mask):
        if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
            or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word input')
        a=self.head;key=a.key(torch.cat((h,roles),-1))
        if self.kind=='rbf12':
            score=-.5*(key[:,None]-a.query[None,:,None]).square().mean(-1)
            alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        else:
            raw=(optical_probability(key,a.query,self.kind)+1e-6)*mask[:,None]
            alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize_head(kind,initial,cfg):
    native=native_initialize('real12',initial,cfg)
    with torch.random.fork_rng(devices=[]):
        model=OpticalHead(kind).to(dtype=getattr(torch,cfg['training_dtype']))
    model.load_state_dict(native.state_dict())
    return model

def validate_initialized_head(model,kind,initial,cfg):
    mapped=dict(cfg,parameter_schemas={'real12':cfg['parameter_schemas'][kind]})
    native_validate(model,'real12',initial,mapped)
