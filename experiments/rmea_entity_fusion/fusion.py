"""Entity partial-swap measurement; known operator, exploratory attention design."""
import math
import torch
from torch import nn
from experiments.rmea_allocation.legacy.rmea import RMEAHead,circuit,EPS

VARIANTS=('global','even','mixture','fusion')

def fuse(a,b,kappa,lam):
    """a,b [...,d,d]; coefficients broadcast before the matrix axes."""
    return ((1-kappa)[...,None,None]*a+(1+kappa)[...,None,None]*b
            +1j*lam[...,None,None]*(a@b-b@a))/2

class FusionHead(nn.Module):
    def __init__(self,variant):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unknown fusion variant')
        self.variant=variant;base=RMEAHead('quantum')
        self.value=base.value;self.decoder=base.decoder;self.bias=base.bias;self.prepare=base.prepare
        if variant=='global':self.effect=base.effect
        elif variant in ('mixture','fusion'):
            self.gates=nn.Parameter(torch.zeros(40,1 if variant=='mixture' else 2))

    def coefficients(self):
        if self.variant in ('global','even'):
            z=self.bias.new_zeros(40);return z,z
        if self.variant=='mixture':return self.gates[:,0].tanh(),self.gates[:,0]*0
        g=self.gates/torch.sqrt(1+self.gates.square().sum(-1,keepdim=True))
        return g[:,0],g[:,1]

    def forward(self,h,roles,mask,details=False,phase_off=False):
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,):raise ValueError('Bad input shapes')
        if mask.dtype!=torch.bool or mask.shape!=h.shape[:2] or not mask.any(-1).all():raise ValueError('Bad mask')
        if h.dtype!=roles.dtype or h.device!=roles.device or h.device!=mask.device:raise ValueError('Bad dtype/device')
        if not torch.isfinite(h).all() or not torch.isfinite(roles).all():raise ValueError('Nonfinite features')
        membership=roles[...,2:4]
        if not ((membership==0)|(membership==1)).all():raise ValueError('Nonbinary entity membership')
        w=membership.transpose(1,2)*mask[:,None,:];counts=w.sum(-1,keepdim=True)
        if not (counts>0).all():raise ValueError('Empty entity after valid mask')
        w=w/counts
        angles=math.pi*torch.tanh(self.prepare(torch.cat((h,roles),-1))).reshape(h.shape[:2]+(2,3,2))
        psi=circuit(angles)
        kappa,lam=self.coefficients()
        if self.variant=='global':
            if phase_off:raise ValueError('No phase-off intervention for global')
            phi=circuit(math.pi*torch.tanh(self.effect))
            probability=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
            raw_probability=probability
        else:
            if phase_off:lam=lam*0
            ab=torch.einsum('bet,bti,btj->beij',w.to(psi.dtype),psi,psi.conj());a,b=ab.unbind(1)
            even=(a+b)/2;diff=(b-a)/2;odd=.5j*(a@b-b@a)
            channel=torch.stack([even,diff,odd],1)
            scores=torch.einsum('bti,bcij,btj->bct',psi.conj(),channel,psi).real
            raw_probability=scores[:,0,None]+kappa[None,:,None]*scores[:,1,None]+lam[None,:,None]*scores[:,2,None]
            # Exact effects are PSD contractions. Clamp floating point roundoff only.
            if raw_probability.min() < -4e-6 or raw_probability.max() > 1+4e-6:raise ValueError('Probability outside rounding tolerance')
            probability=raw_probability.clamp(0,1)
        raw=(probability+EPS)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
        values=self.value(h);utilities=torch.einsum('btd,rd->brt',values,self.decoder)
        logits=(alpha*utilities).sum(-1)+self.bias
        if not details:return logits,alpha
        return dict(logits=logits,alpha=alpha,probability=probability,raw_probability=raw_probability,
                    psi=psi,utilities=utilities,kappa=kappa,lam=lam,
                    clamped=int(((raw_probability<0)|(raw_probability>1)).sum().detach()))
