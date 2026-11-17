"""Task-matched classical controls over the same frozen word representations."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES=('uniform','relation_softmax','real_overlap')


class FrozenFeatureHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown common-feature control')
        self.mode=mode
        # Shared members are initialized before mode-specific members.
        self.value=nn.Linear(1024,64)
        self.decoder=nn.Parameter(torch.randn(40,64)/math.sqrt(64))
        self.bias=nn.Parameter(torch.zeros(40))
        if mode!='uniform':
            self.key=nn.Linear(1028,8)
            self.query=nn.Parameter(torch.randn(40,8)/math.sqrt(8))

    def forward(self,h,roles,mask):
        if h.ndim!=3 or h.shape[-1]!=1024 or roles.shape!=h.shape[:2]+(4,):
            raise ValueError('Expected word states[B,T,1024] and ordered-role features[B,T,4]')
        if mask.shape!=h.shape[:2] or mask.dtype!=torch.bool or not mask.any(-1).all():
            raise ValueError('Each row requires at least one valid original word')
        value=self.value(h)
        if self.mode=='uniform':
            weights=mask[:,None,:].float().expand(-1,40,-1)
            weights=weights/weights.sum(-1,keepdim=True)
        else:
            key=self.key(torch.cat([h,roles],dim=-1))
            if self.mode=='relation_softmax':
                scores=torch.einsum('btd,rd->brt',key,self.query)/math.sqrt(8)
                weights=scores.masked_fill(~mask[:,None,:],-torch.inf).softmax(-1)
            else:
                scores=torch.einsum('btd,rd->brt',F.normalize(key,dim=-1),F.normalize(self.query,dim=-1)).square()
                scores=(scores+1e-6)*mask[:,None,:]
                weights=scores/scores.sum(-1,keepdim=True)
        pooled=torch.einsum('brt,btd->brd',weights,value)
        return (pooled*self.decoder).sum(-1)+self.bias,weights
