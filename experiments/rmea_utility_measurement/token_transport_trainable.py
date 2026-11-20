"""Differentiable fixed-tau token transport, separate from frozen replay code.

The off-diagonal graph is divided by its exact Hermitian operator norm.
No detach, smooth surrogate, fitted scale, or denominator change is used.
At a nonzero spectral tie autograd selects a norm subgradient; a unique
derivative is claimed only off ties. A zero multi-token graph is rejected
because normalized graph transport has no continuous extension there.
"""
import math
import torch
from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead

KINDS=('native','cubic_coherent','cubic_initial_dephased','linear_coherent')
TAU=.25


def route(psi,phi,mask,*,kind,_tau=TAU):
    if kind not in KINDS or not isinstance(_tau,(int,float)) or not math.isfinite(_tau):
        raise ValueError('Unknown fixed routing contract')
    if psi.ndim!=3 or psi.shape[-1]!=8 or min(psi.shape[:2])<1 or phi.shape!=(40,8):
        raise ValueError('Expected [B,T,8] and [40,8]')
    if psi.dtype!=torch.complex128 or phi.dtype!=psi.dtype or phi.device!=psi.device:
        raise ValueError('Training qualification requires complex128')
    if mask.dtype!=torch.bool or mask.shape!=psi.shape[:2] or mask.device!=psi.device or not mask.any(-1).all():
        raise ValueError('Invalid valid-token mask')
    valid=psi[mask]
    if not torch.isfinite(valid).all() or not torch.isfinite(phi).all():
        raise ValueError('Nonfinite valid quantum state')
    if not torch.allclose(valid.norm(dim=-1),torch.ones(len(valid),dtype=torch.float64,device=psi.device),atol=1e-10,rtol=0):
        raise ValueError('Nonunit valid token')
    if not torch.allclose(phi.norm(dim=-1),torch.ones(40,dtype=torch.float64,device=psi.device),atol=1e-10,rtol=0):
        raise ValueError('Nonunit query')
    s=psi.masked_fill(~mask[...,None],0)
    a=torch.einsum('rd,btd->brt',phi,s.conj())
    n=mask.sum(-1)
    denominator=a.abs().square().sum(-1)+n[:,None].to(a.real.dtype)*EPS
    native=(a.abs().square()+EPS)*mask[:,None,:]/denominator[...,None]
    if kind=='native':return dict(alpha=native,native_alpha=native,normalizer=denominator)
    gram=s.conj()@s.transpose(-2,-1)
    graph=gram if kind=='linear_coherent' else gram*gram.abs().square()
    graph=graph-torch.diag_embed(graph.diagonal(dim1=-2,dim2=-1))
    eig=torch.linalg.eigvalsh(graph)
    scale=eig.abs().amax(-1)
    if ((n>1)&(scale==0)).any():raise ValueError('Zero multi-token graph has undefined normalized-transport derivative')
    safe=torch.where(scale>0,scale,torch.ones_like(scale))
    generator=graph/safe[:,None,None]
    unitary=torch.matrix_exp(-1j*float(_tau)*generator)
    if kind=='cubic_initial_dephased':
        alpha=native@unitary.abs().square().transpose(-2,-1)
    else:
        alpha=((a@unitary.transpose(-2,-1)).abs().square()+EPS)/denominator[...,None]
    alpha=alpha*mask[:,None,:]
    if not torch.isfinite(alpha).all():raise ValueError('Nonfinite routed attention')
    return dict(alpha=alpha,native_alpha=native,normalizer=denominator,gram=gram,
                graph=graph,generator=generator,unitary=unitary,spectral_scale=scale,
                eigenvalues=eig,amplitudes=a)


class TrainableTokenTransportHead(RMEAHead):
    """Keep all original 81028 parameters; only routing is replaced."""
    def __init__(self,kind):
        if kind not in KINDS:raise ValueError('Unknown trainable token routing')
        super().__init__('quantum');self.kind=kind

    def forward(self,h,roles,mask,details=False):
        if not torch.isfinite(h).all() or not torch.isfinite(roles).all():
            raise ValueError('Full-head feature and role tensors must be finite, including padding')
        d=super().forward(h,roles,mask,details=True)
        t=route(d['psi'],d['phi'],mask,kind=self.kind)
        alpha=t['alpha'];logits=(alpha*d['utilities']).sum(-1)+self.bias
        if not details:return logits,alpha
        # Inherited raw-score sensitivity is not a derivative through the graph.
        return dict(logits=logits,alpha=alpha,values=d['values'],utilities=d['utilities'],
                    angles=d['angles'],effect_angles=d['effect_angles'],psi=d['psi'],phi=d['phi'],
                    native_alpha=d['alpha'],normalizer=t['normalizer'],transport=t,routing_kind=self.kind)
