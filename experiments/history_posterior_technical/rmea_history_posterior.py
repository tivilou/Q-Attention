"""Nonselective coin-history routing of full original word values.

Scientific plugin only. Forty terminal effects are separate binary tests.
No data, optimizer, scheduler, checkpoint or resource policy is defined here.
"""
import math
import torch
from torch import nn
from baselines.retacred.rmea import circuit
from baselines.retacred.rmea_matched_routers import initialize_head

MODES=('quantum_history','dephased_history','classical_history','gaussian_covariance')
COMMON=('value.weight','value.bias','decoder','bias')


def native_unitary(angles):
    """Pinned native RY/RZ/CNOT order, extended to every basis input."""
    if angles.shape[-3:]!=(2,3,2):raise ValueError('Expected native3q/depth2 angles')
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    state=torch.eye(8,dtype=dtype,device=angles.device).expand(angles.shape[:-3]+(8,8))
    a=angles.unsqueeze(-4)
    for layer in range(2):
        for wire in range(3):
            shape=state.shape
            pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            left,right=pair.unbind(-2)
            y,z=a[...,layer,wire,:].unbind(-1)
            c,s=torch.cos(y/2)[...,None,None],torch.sin(y/2)[...,None,None]
            minus,plus=torch.exp(-.5j*z)[...,None,None],torch.exp(.5j*z)[...,None,None]
            state=torch.stack(((c*left-s*right)*minus,(s*left+c*right)*plus),-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state.transpose(-2,-1)


def valid_inputs(h,roles,mask):
    if (h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,)
            or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all()):
        raise ValueError('Expected full original h1024/roles4 and nonempty valid mask')
    # Sanitize before ALL linear/gate/value operations, including interior holes.
    return torch.where(mask[...,None],h,0),torch.where(mask[...,None],roles,0)


def quantum_masses(U,E,mask):
    batch,length=mask.shape
    rho=U.new_zeros(batch,8,8);rho[:,0,0]=1
    prefixes=[rho]
    for t in range(length):
        proposed=(rho+U[:,t]@rho@U[:,t].mH)/2
        rho=torch.where(mask[:,t,None,None],proposed,rho)
        prefixes.append(rho)
    backward=E[None].expand(batch,-1,-1,-1);masses=[]
    for t in reversed(range(length)):
        u=U[:,t,None]
        selected=u@prefixes[t][:,None]@u.mH/2
        joint=torch.einsum('brij,brji->br',backward,selected).real
        masses.append(torch.where(mask[:,t,None],joint,0))
        before=(backward+u.mH@backward@u)/2
        backward=torch.where(mask[:,t,None,None,None],before,backward)
    return torch.stack(masses[::-1],-1)


def population_masses(P,e,mask):
    """Column-stochastic P; exact sum over classical coin and state histories."""
    batch,length=mask.shape
    mass=P.new_zeros(batch,8);mass[:,0]=1
    prefixes=[mass]
    for t in range(length):
        proposed=(mass+(P[:,t]@mass[...,None])[...,0])/2
        mass=torch.where(mask[:,t,None],proposed,mass);prefixes.append(mass)
    backward=e[None].expand(batch,-1,-1);masses=[]
    for t in reversed(range(length)):
        selected=(P[:,t]@prefixes[t][...,None])[...,0]/2
        joint=torch.einsum('bri,bi->br',backward,selected)
        masses.append(torch.where(mask[:,t,None],joint,0))
        before=(backward+torch.einsum('brj,bji->bri',backward,P[:,t]))/2
        backward=torch.where(mask[:,t,None,None],before,backward)
    return torch.stack(masses[::-1],-1)


class HistoryHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES[:3]:raise ValueError('Unknown history mode')
        self.mode=mode
        self.value=nn.Linear(1024,64)
        self.decoder=nn.Parameter(torch.randn(40,64)/math.sqrt(64))
        self.bias=nn.Parameter(torch.zeros(40))
        self.prepare=nn.Linear(1028,12)
        if mode=='classical_history':
            self.transition=nn.Linear(12,64)
            self.terminal=nn.Parameter(torch.randn(40,8)*.2)
        else:self.effect=nn.Parameter(torch.randn(40,2,3,2)*.2)

    def forward(self,h,roles,mask,details=False,same_state_dephased=False):
        h,roles=valid_inputs(h,roles,mask)
        z=torch.tanh(self.prepare(torch.cat((h,roles),-1)))
        if self.mode=='classical_history':
            if same_state_dephased:raise ValueError('Quantum intervention requires Q')
            P=self.transition(z).reshape(h.shape[:2]+(8,8)).softmax(-2)
            e=(self.terminal.softmax(-1)+1/8)/2
            masses=population_masses(P,e,mask)
        else:
            U=native_unitary((math.pi*z).reshape(h.shape[:2]+(2,3,2)))
            phi=circuit(math.pi*torch.tanh(self.effect))
            I=torch.eye(8,dtype=U.dtype,device=U.device)
            E=(phi[:,:,None]*phi[:,None,:].conj()+I[None]/8)/2
            if self.mode=='dephased_history' or same_state_dephased:
                # Δ after every selected/nonselective map INCLUDING the last.
                masses=population_masses(U.abs().square(),E.diagonal(dim1=-2,dim2=-1).real,mask)
            else:masses=quantum_masses(U,E,mask)
        alpha=masses/masses.sum(-1,keepdim=True)
        values=torch.where(mask[...,None],self.value(h),0)
        logits=torch.einsum('brt,btd,rd->br',alpha,values,self.decoder)+self.bias
        if details:return dict(logits=logits,alpha=alpha,masses=masses,values=values)
        return logits

    def same_state_dephased_logits(self,h,roles,mask):
        if self.mode!='quantum_history':raise ValueError('Requires fixed Q state')
        return self(h,roles,mask,same_state_dephased=True)


class GaussianCovarianceHead(nn.Module):
    """Exact accepted Gcov formula/initialization; only lift root key names."""
    def __init__(self):
        super().__init__()
        self.value=nn.Linear(1024,64)
        self.decoder=nn.Parameter(torch.randn(40,64)/math.sqrt(64))
        self.bias=nn.Parameter(torch.zeros(40))
        self.key=nn.Linear(1028,12)
        self.query=nn.Parameter(torch.randn(40,12)/math.sqrt(12))
        self.moment_decoder=nn.Parameter(torch.zeros(40,32))

    def forward(self,h,roles,mask,details=False):
        h,roles=valid_inputs(h,roles,mask)
        key=self.key(torch.cat((h,roles),-1))
        scale=(key.new_tensor([3-math.sqrt(2),3+math.sqrt(2)]*6)/24).sqrt()
        k,q=key*scale,self.query*scale
        raw=(torch.exp(-(k[:,None]-q[None,:,None]).square().sum(-1))+1e-6)*mask[:,None]
        alpha=raw/raw.sum(-1,keepdim=True)
        values=torch.where(mask[...,None],self.value(h),0)
        mean=torch.einsum('brt,btd->brd',alpha,values)
        connected=torch.einsum('brt,btj->brj',alpha,values[...,0::2]*values[...,1::2])-mean[...,0::2]*mean[...,1::2]
        logits=(mean*self.decoder).sum(-1)+self.bias+(connected*self.moment_decoder).sum(-1)
        if details:return dict(logits=logits,alpha=alpha,values=values,connected=connected)
        return logits


def make_models(seed=13,initial=None):
    if initial is None:raise ValueError('Saved/pinned native initialization required')
    result={}
    with torch.random.fork_rng(devices=[]):
        for mode in MODES:
            torch.manual_seed(seed)
            model=GaussianCovarianceHead() if mode=='gaussian_covariance' else HistoryHead(mode)
            state=model.state_dict()
            if mode in MODES[:2]:state={k:initial[k].detach().clone() for k in state}
            elif mode=='classical_history':
                for k in COMMON+('prepare.weight','prepare.bias'):state[k]=initial[k].detach().clone()
            else:
                legacy=initialize_head('real12',initial,dict(seed=seed,training_dtype='float32')).head.state_dict()
                for k in legacy:state[k]=legacy[k].detach().clone()
            model.load_state_dict(state,strict=True);result[mode]=model
    return result


def real_parameters(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
