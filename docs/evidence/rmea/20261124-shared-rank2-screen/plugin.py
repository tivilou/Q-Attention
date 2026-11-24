"""Experimental shared measurement mixture; no training/storage/scheduler logic.

Not yet executed or admitted. The wrapper preserves the complete native module
and adds only an independent second query. K=1 provides an affected parity path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit, EPS


class SharedMeasurement(nn.Module):
    def __init__(self, native, components=2, extra_seed=9200):
        super().__init__()
        if native.mode not in ('quantum','dephased','real12','softmax12'):
            raise ValueError('Unsupported native condition')
        if components not in (1,2):raise ValueError('Only parity K1 and frozen K2 supported')
        self.native=native;self.components=components
        a=native.head;self.mode=native.mode
        if components==2:
            reference=a.effect if self.mode in ('quantum','dephased') else a.query
            # Construct before moving model to CUDA, preserving the caller's RNG.
            if reference.device.type!='cpu':raise ValueError('Initialize extra query on CPU')
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(extra_seed)
                extra=torch.randn(reference.shape,dtype=reference.dtype)*(.2 if self.mode in ('quantum','dephased') else 1/math.sqrt(12))
            self.extra=nn.Parameter(extra)

    @property
    def head(self):
        return self.native.head

    def forward(self,h,roles,mask):
        if self.components==1:return self.native(h,roles,mask)
        a=self.native.head
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,):
            raise ValueError('Invalid frozen word/role input')
        if mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():
            raise ValueError('Invalid word mask')
        x=torch.cat((h,roles),-1)
        if self.mode in ('quantum','dephased'):
            key=circuit((math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(2,3,2)))
            query=circuit(math.pi*torch.tanh(torch.stack((a.effect,self.extra),1)))
            if self.mode=='quantum':
                score=torch.einsum('rkd,btd->brkt',query.conj(),key).abs().square().mean(2)
            else:
                score=torch.einsum('rkd,btd->brkt',query.abs().square(),key.abs().square()).mean(2)
        else:
            key=a.key(x);query=torch.stack((a.query,self.extra),1)
            if self.mode=='real12':
                score=torch.einsum('rkd,btd->brkt',F.normalize(query,dim=-1),F.normalize(key,dim=-1)).square().mean(2)
            else:
                components=torch.einsum('rkd,btd->brkt',query,key)/math.sqrt(12)
                score=torch.logsumexp(components,dim=2)-math.log(2)
        if self.mode=='softmax12':
            alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        else:
            raw=(score+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
        utilities=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
        return (alpha*utilities).sum(-1)+a.bias
