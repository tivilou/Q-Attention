"""Learned positive measurement spectrum on the original RMEA circuit."""
import math
import torch
from torch import nn
from legacy.rmea import RMEAHead, EPS

VARIANTS=('static','spectral','scalar_calibration','spectral_dephased')
INITIAL_SPECTRUM=(.9,)+(.1/7,)*7
CALIBRATION_VALUES=tuple(INITIAL_SPECTRUM[1]+(INITIAL_SPECTRUM[0]-INITIAL_SPECTRUM[1])*k/7 for k in range(8))
CALIBRATION_SCALE=sum(CALIBRATION_VALUES)


def circuit_columns(angles):
    """Rows k are U|k>; same RY/RZ/CNOT gates, wire0 MSB, as legacy circuit."""
    if angles.shape[-3:]!=(2,3,2):raise ValueError('Expected [...,2,3,2]')
    batch=angles.shape[:-3]
    angles=angles.unsqueeze(-4).expand(batch+(8,2,3,2))
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    state=torch.eye(8,device=angles.device,dtype=dtype).expand(batch+(8,8))
    for layer in range(2):
        for wire in range(3):
            shape=state.shape;pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=pair.unbind(-2);y,z=angles[...,layer,wire,:].unbind(-1)
            c=torch.cos(y/2)[...,None,None];s=torch.sin(y/2)[...,None,None]
            minus=torch.exp(-.5j*z)[...,None,None];plus=torch.exp(.5j*z)[...,None,None]
            state=torch.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),dim=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state


def spectral_probability(psi,basis,spectrum,dephased=False):
    if dephased:
        channels=torch.einsum('rkd,btd->brtk',basis.abs().square(),psi.abs().square())
    else:
        channels=torch.einsum('rkd,btd->brtk',basis.conj(),psi).abs().square()
    return torch.einsum('brtk,rk->brt',channels,spectrum),channels


def scalar_probability(p,coefficients):
    # A positive Bernstein affinity, not a Born probability. Its fixed scale
    # makes its initial FUNCTION (including EPS) match the spectral condition.
    p=p.clamp(0,1)
    terms=torch.stack([math.comb(7,k)*p.pow(k)*(1-p).pow(7-k) for k in range(8)],-1)
    return CALIBRATION_SCALE*torch.einsum('brtk,rk->brt',terms,coefficients)


class SpectrumHead(RMEAHead):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('Unknown variant')
        super().__init__('quantum');self.variant=variant
        if variant!='static':
            initial=CALIBRATION_VALUES if variant=='scalar_calibration' else INITIAL_SPECTRUM
            self.spectrum_logits=nn.Parameter(torch.tensor(initial).log().repeat(40,1))

    def forward(self,h,roles,mask,details=False):
        old=super().forward(h,roles,mask,details=True)
        if self.variant=='static':return old if details else (old['logits'],old['alpha'])
        spectrum=self.spectrum_logits.softmax(-1)
        if self.variant=='scalar_calibration':
            probability=scalar_probability(old['probability'],spectrum)
            extra=dict(calibration=spectrum,original_probability=old['probability'])
        else:
            basis=circuit_columns(old['effect_angles'])
            probability,channels=spectral_probability(old['psi'],basis,spectrum,self.variant=='spectral_dephased')
            extra=dict(basis=basis,spectrum=spectrum,channels=channels)
        raw=(probability+EPS)*mask[:,None,:];normalizer=raw.sum(-1,keepdim=True);alpha=raw/normalizer
        logits=(alpha*old['utilities']).sum(-1)+self.bias
        if not details:return logits,alpha
        return dict(logits=logits,alpha=alpha,values=old['values'],utilities=old['utilities'],
            angles=old['angles'],effect_angles=old['effect_angles'],psi=old['psi'],phi=old['phi'],
            probability=probability,normalizer=normalizer,**extra)
