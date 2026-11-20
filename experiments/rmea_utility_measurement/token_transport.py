"""Fixed-checkpoint token-graph routing; no training or fitted transport parameter."""
import math
import torch
from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
KINDS=('native','coherent','token_dephased')
TAU=.25

@torch.no_grad()
def transport_attention(psi,phi,mask,*,kind='coherent',_tau=TAU):
    if kind not in KINDS or not isinstance(_tau,(int,float)) or not math.isfinite(_tau):raise ValueError('Unknown fixed transport interface')
    if psi.ndim!=3 or psi.shape[-1]!=8 or psi.shape[0]<1 or psi.shape[1]<1 or phi.shape!=(40,8):raise ValueError('Expected [B,T,8] and [40,8]')
    if psi.dtype!=torch.complex128 or phi.dtype!=psi.dtype or psi.device!=phi.device:raise ValueError('Forward qualification requires complex128')
    if mask.dtype!=torch.bool or mask.shape!=psi.shape[:2] or mask.device!=psi.device or not mask.any(-1).all():raise ValueError('Invalid valid-token mask')
    if not torch.isfinite(phi).all() or not torch.allclose(phi.norm(dim=-1),torch.ones(40,dtype=torch.float64,device=phi.device),atol=1e-10,rtol=0):raise ValueError('Nonunit query')
    out=torch.zeros((psi.shape[0],40,psi.shape[1]),dtype=torch.float64,device=psi.device);samples=[]
    for b in range(len(psi)):
        ix=torch.where(mask[b])[0];s=psi[b,ix];n=len(ix)
        if not torch.isfinite(s).all() or not torch.allclose(s.norm(dim=-1),torch.ones(n,dtype=torch.float64,device=s.device),atol=1e-10,rtol=0):raise ValueError('Nonunit token')
        a=phi@s.conj().T;D=a.abs().square().sum(-1)+n*EPS
        native=(a.abs().square()+EPS)/D[:,None]
        G=s.conj()@s.T;K=G*G.abs().square();K=K-torch.diag_embed(K.diagonal())
        scale=torch.linalg.eigvalsh(K).abs().max()
        H=K/scale if float(scale)>0 else torch.zeros_like(K)
        U=torch.matrix_exp(-1j*float(_tau)*H)
        if kind=='native':alpha=native
        elif kind=='coherent':alpha=((a@U.T).abs().square()+EPS)/D[:,None]
        else:alpha=native@U.abs().square().T
        assert torch.isfinite(alpha).all()
        out[b].index_copy_(1,ix,alpha)
        samples.append(dict(valid_indices=ix,states=s,a=a,D=D,G=G,K=K,H=H,U=U,native_alpha=native,alpha=alpha))
    return dict(alpha=out,samples=samples)

class TokenTransportHead(RMEAHead):
    def __init__(self,kind):
        if kind not in KINDS:raise ValueError('Unknown transport kind')
        super().__init__('quantum');self.kind=kind

    @torch.no_grad()
    def forward(self,h,roles,mask,details=False):
        d=super().forward(h,roles,mask,details=True)
        if self.kind=='native':return d if details else (d['logits'],d['alpha'])
        t=transport_attention(d['psi'],d['phi'],mask,kind=self.kind)
        alpha=t['alpha'];logits=(alpha*d['utilities']).sum(-1)+self.bias
        if not details:return logits,alpha
        return dict(d,alpha=alpha,logits=logits,native_alpha=d['alpha'],transport=t)
