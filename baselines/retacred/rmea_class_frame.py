"""Class-axis square-root measurement; weighted utility readout stays unchanged.

The PGM construction is known. This isolated head is a candidate routing change,
not a task-utility, quantum-exclusivity, or novelty certificate.
"""
import torch
from .rmea import EPS,RMEAHead

KINDS=('native','class_frame','class_frame_diagonal')

def hermitian(x):
    return (x+x.mH)/2

class _InverseSqrt(torch.autograd.Function):
    """First derivative uses the continuous divided difference at repeated roots."""
    @staticmethod
    def forward(ctx,h):
        ev,v=torch.linalg.eigh(hermitian(h))
        if not torch.isfinite(ev).all() or (ev<=0).any():
            raise ValueError('Strictly positive finite class frame required')
        root=ev.sqrt()
        ctx.save_for_backward(root,v)
        return hermitian((v/root.unsqueeze(-2))@v.mH)

    @staticmethod
    def backward(ctx,grad):
        root,v=ctx.saved_tensors
        g=v.mH@hermitian(grad)@v
        denom=root.unsqueeze(-1)*root.unsqueeze(-2)*(root.unsqueeze(-1)+root.unsqueeze(-2))
        return hermitian(v@(-g/denom)@v.mH)

def inverse_sqrt(h):
    if h.ndim<2 or h.shape[-1]!=h.shape[-2] or not torch.isfinite(h).all():
        raise ValueError('Finite square matrix required')
    if not torch.allclose(h,h.mH,rtol=0,atol=1e-10):
        raise ValueError('Hermitian frame required')
    return _InverseSqrt.apply(h)

def route(psi,phi,mask,*,kind):
    if kind not in KINDS or psi.ndim!=3 or psi.shape[-1]!=8 or phi.shape!=(40,8):
        raise ValueError('Expected explicit kind, [B,T,8] states and40queries')
    if psi.dtype!=torch.complex128 or phi.dtype!=psi.dtype or phi.device!=psi.device:
        raise ValueError('Complex128 on one device required')
    if mask.dtype!=torch.bool or mask.shape!=psi.shape[:2] or not mask.any(-1).all():
        raise ValueError('Nonempty valid-token mask required')
    if not torch.isfinite(psi).all() or not torch.isfinite(phi).all():
        raise ValueError('Finite states including padding required')
    if not torch.allclose(psi[mask].norm(dim=-1),torch.ones_like(psi[mask].real[:,0]),rtol=0,atol=1e-10):
        raise ValueError('Unit token states required')
    if not torch.allclose(phi.norm(dim=-1),torch.ones(40,dtype=torch.float64,device=phi.device),rtol=0,atol=1e-10):
        raise ValueError('Unit class states required')
    eye=torch.eye(8,dtype=phi.dtype,device=phi.device)
    frame=phi.T@phi.conj()+40*EPS*eye
    if kind=='native':
        raw=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()+EPS
        w=eye
    else:
        if kind=='class_frame_diagonal':
            # Basis-dependent control. It is positive, but is generally NOT a POVM.
            w=torch.diag(frame.diagonal().real.rsqrt()).to(phi.dtype)
        else:
            w=inverse_sqrt(frame)
        transformed=torch.einsum('de,bte->btd',w,psi)
        raw=torch.einsum('rd,btd->brt',phi.conj(),transformed).abs().square()
        raw=raw+EPS*transformed.abs().square().sum(-1)[:,None,:]
    raw=raw*mask[:,None,:]
    normalizer=raw.sum(-1,keepdim=True)
    if not torch.isfinite(raw).all() or (normalizer<=0).any():
        raise ValueError('Finite positive measurement denominator required')
    alpha=raw/normalizer
    return dict(alpha=alpha,raw=raw,normalizer=normalizer,frame=frame,inverse_sqrt=w,kind=kind)

class ClassFrameHead(RMEAHead):
    """Same state_dict and81028 trainable parameters; no learned routing extra."""
    def __init__(self,kind):
        if kind not in KINDS:raise ValueError('Unknown class-frame kind')
        super().__init__('quantum');self.kind=kind

    def forward(self,h,roles,mask,details=False):
        if not torch.isfinite(h).all() or not torch.isfinite(roles).all():
            raise ValueError('Finite features including padding required')
        d=super().forward(h,roles,mask,details=True)
        if self.kind=='native':
            if details:return dict(d,routing_kind=self.kind)
            return d['logits'],d['alpha']
        t=route(d['psi'],d['phi'],mask,kind=self.kind)
        alpha=t['alpha'];logits=(alpha*d['utilities']).sum(-1)+self.bias
        if not details:return logits,alpha
        return dict(logits=logits,alpha=alpha,native_alpha=d['alpha'],values=d['values'],utilities=d['utilities'],psi=d['psi'],phi=d['phi'],angles=d['angles'],effect_angles=d['effect_angles'],measurement=t,routing_kind=self.kind)
