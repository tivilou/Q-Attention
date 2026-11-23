"""Two returns through the relative encoding unitary; no trainable parameters added."""
import math
import torch
from torch import nn


def unitary_from_native(circuit,angles):
    """Exact columns via the existing RY/RZ circuit, wire0 MSB.

    RY(a+pi*b)|0> = RY(a)|b> for b in{0,1}, with no basis-dependent phase.
    The additional pi is basis-column evaluation, not a learned gate angle.
    """
    if angles.shape[-3:]!=(2,3,2):raise ValueError('Native physical angles required')
    columns=angles.unsqueeze(-4).expand(angles.shape[:-3]+(8,2,3,2)).clone()
    bits=((torch.arange(8,device=angles.device)[:,None]>>torch.tensor([2,1,0],device=angles.device))&1).to(angles.dtype)
    columns[...,0,:,0]=columns[...,0,:,0]+math.pi*bits
    return circuit(columns).transpose(-1,-2)


def return_scores(circuit,token_angles,class_angles):
    U=unitary_from_native(circuit,token_angles);V=unitary_from_native(circuit,class_angles)
    psi=U[...,0];phi=V[...,0]
    # a_j=W_j0, b_j=W_0j for W=V^dagger U. No B,R,T,8,8 tensor.
    a=torch.einsum('rkj,btk->brtj',V.conj(),psi)
    b=torch.einsum('rk,btkj->brtj',phi.conj(),U)
    paths=a*b
    return dict(coherent=paths.sum(-1).abs().square(),dephased=paths.abs().square().sum(-1),
                native=a[...,0].abs().square(),token_unitary=U,class_unitary=V,paths=paths)


class RelativeReturnHead(nn.Module):
    """Scientific wrapper only: unchanged features, value, decoder and native parameters."""
    def __init__(self,native,circuit,mode,eps=1e-6):
        super().__init__()
        if mode not in ('coherent','dephased'):raise ValueError('Unknown readout')
        self.native=native;self.circuit=circuit;self.mode=mode;self.eps=eps
    def forward(self,h,roles,mask,details=False):
        d=self.native(h,roles,mask,details=True)
        r=return_scores(self.circuit,d['angles'],d['effect_angles']);p=r[self.mode]
        raw=(p+self.eps)*mask[:,None];normalizer=raw.sum(-1,keepdim=True);alpha=raw/normalizer
        logits=(alpha*d['utilities']).sum(-1)+self.native.bias
        if details:return dict(logits=logits,alpha=alpha,probability=p,normalizer=normalizer,native=d,returns=r)
        return logits,alpha
