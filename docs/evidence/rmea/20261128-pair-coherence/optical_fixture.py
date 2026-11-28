"""Differentiable synthetic front end for the fixed optical resource and its two controls."""
import hashlib
import json
import math
from pathlib import Path

import torch
import numpy as np
from torch import nn

from pair_coherence import kernels as reference

MODES=('coherent','number_diagonal','curvature_gaussian')


def probabilities(key, query, mode):
    # Same 12 original real coordinates -> six complex displacements.
    d=(key[:,None]-query[None,:,None]).reshape(key.shape[0],40,key.shape[1],6,2)/math.sqrt(24)
    re,im=d.unbind(-1)
    a,b=re[...,:3],im[...,:3]
    active=a.square()+b.square()
    envelope=(re.square()+im.square()).sum(-1)
    if mode=='coherent':
        re_sum=(a.square()-b.square()).sum(-1)
        im_sum=2*(a*b).sum(-1)
        overlap=(1-2*active.mean(-1)+(re_sum.square()+im_sum.square())/6)
        return (overlap*torch.exp(-envelope/2)).square()
    if mode=='number_diagonal':
        laguerre=1-2*active+active.square()/2
        g0,g1,g2=active.square().unbind(-1)
        scale=(laguerre.square().sum(-1)+(g0*g1+g1*g2+g0*g2)/2)/3
        return scale*torch.exp(-envelope)
    if mode=='curvature_gaussian':
        return torch.exp(-envelope-4*active.sum(-1)/3)
    raise ValueError(mode)


class FixtureHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode=mode
        self.key=nn.Linear(1028,12)
        self.query=nn.Parameter(torch.randn(40,12)*.3)
        self.value=nn.Linear(1024,64)
        self.decoder=nn.Parameter(torch.randn(40,64)/8)
        self.bias=nn.Parameter(torch.zeros(40))

    def forward(self,h,roles,mask,details=False):
        assert h.shape[-1]==1024 and roles.shape==(*h.shape[:2],4)
        assert mask.shape==h.shape[:2] and mask.dtype==torch.bool and mask.any(-1).all()
        h=h.masked_fill(~mask[...,None],0)
        r=roles.masked_fill(~mask[...,None],0)
        weights=(probabilities(self.key(torch.cat((h,r),-1)),self.query,self.mode)+1e-6)*mask[:,None]
        alpha=weights/weights.sum(-1,keepdim=True)
        v=self.value(h)
        pooled=torch.einsum('brt,btd->brd',alpha,v)
        logits=(pooled*self.decoder).sum(-1)+self.bias
        return (logits,alpha,pooled) if details else logits


def test():
    torch.manual_seed(20261128)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    h=torch.randn(3,10,1024)
    r=torch.randn(3,10,4)
    mask=(torch.arange(10)[None,:]<torch.tensor([1,7,10])[:,None])
    mask[1,2]=False
    y=torch.tensor([0,21,39])
    initial=FixtureHead(MODES[0]).state_dict()
    checks={}
    for i,mode in enumerate(MODES):
        m=FixtureHead(mode)
        m.load_state_dict(initial,strict=True)
        z,alpha,pool=m(h,r,mask,details=True)
        z2=m(h.masked_fill(~mask[...,None],float('nan')),
             r.masked_fill(~mask[...,None],float('nan')),mask)
        assert torch.equal(z,z2) and z.shape==(3,40) and pool.shape==(3,40,64)
        loss=torch.nn.functional.cross_entropy(z,y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters())
        with torch.no_grad():
            masked_h=h.masked_fill(~mask[...,None],0)
            masked_r=r.masked_fill(~mask[...,None],0)
            k=m.key(torch.cat((masked_h,masked_r),-1)).numpy()
            q=m.query.numpy()
            delta=(k[:,None]-q[None,:,None])/math.sqrt(24)
            d=delta[...,0::2]+1j*delta[...,1::2]
            w=(reference(d)[...,i]+1e-6)*mask.numpy()[:,None,:]
            ahat=w/w.sum(-1,keepdims=True)
            values=m.value(masked_h).numpy()
            vhat=ahat@values
            zhat=(vhat*m.decoder.numpy()).sum(-1)+m.bias.numpy()
            err=float(np.max(np.abs(zhat-z.numpy())))
            assert err<1e-12 and np.max(np.abs(vhat-pool.numpy()))<1e-12
            maximum=0.
            step=1e-5
            for p,index in [(m.key.weight,(1,0)),(m.query,(21,2)),(m.value.weight,(3,42)),(m.decoder,(21,3))]:
                initial_value=p[index].item()
                p[index]=initial_value+step
                plus=torch.nn.functional.cross_entropy(m(h,r,mask),y).item()
                p[index]=initial_value-step
                minus=torch.nn.functional.cross_entropy(m(h,r,mask),y).item()
                p[index]=initial_value
                maximum=max(maximum,abs((plus-minus)/(2*step)-p.grad[index].item()))
            assert maximum<1e-8
            # FP32 numerical consistency on the same model and input, without training.
            fp32=m.float()(h.float(),r.float(),mask)
            error32=float((z-fp32).abs().max())
            assert error32<5e-6
        checks[mode]=dict(full_logits_error=err,gradient_max_fd_error=maximum,fp32_logit_delta=error32)
    return dict(status='PASS_SYNTHETIC_FULL_VALUE_AND_GRADIENTS',arms=checks,all40=True,full_values=64,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input='three artificial 1024+4 word sequences including interior missing mask and one word',
                task_forwards=0,optimizer_updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(test(),indent=2,allow_nan=False))