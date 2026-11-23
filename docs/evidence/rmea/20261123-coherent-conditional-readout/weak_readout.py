"""Word-register coherent conditional readout; stable trainer interface only."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import RMEAHead
from baselines.retacred.frozen_feature_head import FrozenFeatureHead

MODES=('coherent','diagonal','classical','native')
EPS=1e-6


def signed_weights(psi,phi,mask,diagonal=False):
    """Real weak conditional word coefficients; an EPS identity-noise branch."""
    psi=psi*mask[...,None]
    if diagonal:
        total=psi.sum(1)
        per_basis=(total[:,None,:].conj()*psi).real
        numerator=torch.einsum('rd,btd->brt',phi.abs().square(),per_basis)
        denominator=torch.einsum('rd,bd->br',phi.abs().square(),total.abs().square())
    else:
        amplitude=torch.einsum('rd,btd->brt',phi.conj(),psi)
        total=amplitude.sum(-1,keepdim=True)
        numerator=(total.conj()*amplitude).real
        denominator=total.abs().square().squeeze(-1)
    noise=mask.to(numerator.dtype)*EPS
    numerator=numerator+noise[:,None,:]
    return numerator/(denominator+noise.sum(-1,keepdim=True))[...,None]


class ConditionalHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown readout condition')
        self.mode=mode
        self.head=FrozenFeatureHead('real_overlap') if mode=='classical' else RMEAHead('quantum')

    def forward(self,h,roles,mask,details=False):
        if mask.dtype!=torch.bool or mask.shape!=h.shape[:2] or not mask.any(-1).all():raise ValueError('Nonempty valid word mask required')
        if self.mode=='classical':
            psi=F.normalize(self.head.key(torch.cat((h,roles),-1)),dim=-1)
            phi=F.normalize(self.head.query,dim=-1)
            values=self.head.value(h)
            utilities=torch.einsum('btd,rd->brt',values,self.head.decoder)
            weights=signed_weights(psi,phi,mask)
            logits=(weights*utilities).sum(-1)+self.head.bias
        else:
            d=self.head(h,roles,mask,details=True)
            psi,phi,utilities=d['psi'],d['phi'],d['utilities']
            weights=d['alpha'] if self.mode=='native' else signed_weights(psi,phi,mask,self.mode=='diagonal')
            logits=d['logits'] if self.mode=='native' else (weights*utilities).sum(-1)+self.head.bias
        if details:return dict(logits=logits,weights=weights,utilities=utilities,psi=psi,phi=phi)
        return logits


def initialize(initial,seed):
    heads={}
    for mode in MODES:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=ConditionalHead(mode).double()
        if mode=='classical':
            state=model.head.state_dict()
            for key in ('value.weight','value.bias','decoder','bias'):state[key]=initial[key].clone()
            model.head.load_state_dict(state)
        else:model.head.load_state_dict(initial)
        heads[mode]=model
    return heads


def fixtures(models,inputs):
    """New readout only: equations, mask, gradient parity, known convex escape."""
    errors={}
    def check(k,value):
        v=float(value)
        if not math.isfinite(v):raise ValueError('Nonfinite fixture '+k)
        errors[k]=v
    for mode,source in models.items():
        model=ConditionalHead(mode).double().to(inputs[0].device);model.load_state_dict(source.state_dict())
        d=model(*inputs,details=True);w=d['weights'];u=d['utilities']
        check(mode+'_sum_one',(w.sum(-1)-1).abs().max())
        direct=(w*u).sum(-1)+model.head.bias
        check(mode+'_formula',(direct-d['logits']).abs().max())
        shifted=(w*(u+7.25)).sum(-1)+model.head.bias
        check(mode+'_constant_preserved',(shifted-d['logits']-7.25).abs().max())
        g1=torch.autograd.grad(d['logits'].square().mean(),tuple(model.parameters()),retain_graph=True)
        g2=torch.autograd.grad(direct.square().mean(),tuple(model.parameters()))
        for j,(x,y) in enumerate(zip(g1,g2)):check(mode+'_gradient_'+str(j),(x-y).abs().max())
        from torch.nn.functional import pad
        pp=(pad(inputs[0],(0,0,0,2),value=.123),pad(inputs[1],(0,0,0,2),value=1.),pad(inputs[2],(0,2),value=False))
        check(mode+'_padding',(model(*pp)-d['logits']).abs().max())
        # Independently construct the token x internal density matrix for one input.
        ids=torch.where(inputs[2][0])[0];t=len(ids);dimension=d['psi'].shape[-1]
        ps=d['psi'][0,ids].detach().to(torch.complex128);ph=d['phi'][1].detach().to(torch.complex128)
        pre=ps.reshape(-1)/math.sqrt(t)
        rho=pre[:,None]*pre.conj()[None,:]+(EPS/t)*torch.eye(t*dimension,dtype=pre.dtype,device=pre.device)
        if mode=='native':
            rho=(rho.reshape(t,dimension,t,dimension)*torch.eye(t,dtype=pre.dtype,device=pre.device)[:,None,:,None]).reshape(t*dimension,t*dimension)
        elif mode=='diagonal':
            rho=(rho.reshape(t,dimension,t,dimension)*torch.eye(dimension,dtype=pre.dtype,device=pre.device)[None,:,None,:]).reshape(t*dimension,t*dimension)
        post=ph.repeat(t)/math.sqrt(t)
        observable=u[0,1,ids].detach().repeat_interleave(dimension)
        denominator=(post.conj()@(rho@post)).real
        physical=(post.conj()@(observable*(rho@post))).real/denominator
        check(mode+'_density_matrix',(physical-(d['logits'][0,1]-model.head.bias[1])).abs())
    # Known two-word signed example, embedded into the actual8-dimensional state space.
    psi=torch.zeros((1,2,8),dtype=torch.complex128,device=inputs[0].device)
    psi[0,0,0]=1;psi[0,1,0]=-.5;psi[0,1,1]=math.sqrt(.75)
    phi=torch.zeros((1,8),dtype=torch.complex128,device=psi.device);phi[0,0]=1
    mask=torch.ones((1,2),dtype=torch.bool,device=psi.device)
    beta=signed_weights(psi,phi,mask)
    expected=(.5+EPS)/(.25+2*EPS)
    check('known_two_word_anomaly',(beta[0,0,0]-expected).abs())
    assert float(beta[0,0,0])>1 and float(beta[0,0,1])<0
    cancellation=torch.zeros((1,3,8),dtype=torch.complex128,device=psi.device)
    cancellation[0,:2,0]=.5;cancellation[0,2,0]=-1
    cancellation[0,0,1]=math.sqrt(.75);cancellation[0,1,1]=-math.sqrt(.75)
    cancelmask=torch.ones((1,3),dtype=torch.bool,device=psi.device)
    for diagonal in (False,True):
        w=signed_weights(cancellation,phi,cancelmask,diagonal)
        check('cancellation_uniform_'+str(diagonal),(w-1/3).abs().max())
    if max(errors.values())>=1e-8:raise ValueError('Readout fixture failed')
    return dict(status='PASS',errors=errors,scope='Only new conditional readout; no repeated lifecycle/COS matrix',
                known_toy_positive_weight=float(beta[0,0,0]),known_toy_negative_weight=float(beta[0,0,1]))
