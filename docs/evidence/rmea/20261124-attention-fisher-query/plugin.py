"""Query-step direction correction using the actual normalized-routing Fisher.

This is norm-matched Fisher-shaped AdamW, NOT coordinate-invariant natural
gradient. No forward modification, learnable additions, runtime or storage code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head

MODES=('quantum_adam','quantum_fisher','dephased_fisher','real12_fisher','softmax12_fisher')
BASE=dict(zip(MODES,('quantum','quantum','dephased','real12','softmax12')))
DAMPING_REL=.01
DAMPING_ABS=1e-8

def query_parameter(model):
    return model.head.effect if model.base in ('quantum','dephased') else model.head.query

@torch.no_grad()
def routing_metric(model,inputs,*,jacobian=False):
    """Mean over input examples of each row's categorical-attention Fisher.

Raw log-response derivatives are centered AFTER the exact word normalization.
Query-only block, with keys fixed at this current forward; no labels enter F.
"""
    h,roles,mask=inputs;a=model.head;x=torch.cat((h,roles),-1)
    if model.base in ('quantum','dephased'):
        key=circuit((math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(2,3,2)))
        angles=math.pi*torch.tanh(a.effect);q=circuit(angles)
        shifted=angles.reshape(40,1,12)+math.pi*torch.eye(12,device=h.device,dtype=h.dtype)[None]
        # d R(theta)/d theta = R(theta+pi)/2 for either native Pauli gate.
        dq=circuit(shifted.reshape(40,12,2,3,2))
        dq=dq*(math.pi/2*(1-torch.tanh(a.effect).square())).reshape(40,12,1)
        if model.base=='quantum':
            amplitude=torch.einsum('rd,btd->brt',q.conj(),key)
            derivative=torch.einsum('rjd,btd->brtj',dq.conj(),key)
            p=amplitude.abs().square()
            dp=2*(amplitude.conj()[...,None]*derivative).real
        else:
            p=torch.einsum('rd,btd->brt',q.abs().square(),key.abs().square())
            dp=torch.einsum('rjd,btd->brtj',2*(q.conj()[:,None]*dq).real,key.abs().square())
        raw=(p.double()+EPS)*mask[:,None]
        alpha=raw/raw.sum(-1,keepdim=True)
        g=dp.double()/(p.double()+EPS)[...,None]
    else:
        key=a.key(x)
        if model.base=='softmax12':
            score=torch.einsum('btd,rd->brt',key,a.query)/math.sqrt(12)
            alpha=score.double().masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            g=(key.double()/math.sqrt(12))[:,None].expand(-1,40,-1,-1)
        else:
            kn=F.normalize(key,dim=-1);qn=F.normalize(a.query,dim=-1)
            overlap=torch.einsum('btd,rd->brt',kn,qn)
            p=overlap.square()
            norm=a.query.norm(dim=-1)
            projected=kn[:,None]-overlap[...,None]*qn[None,:,None]
            tangent=torch.where((norm>=1e-12)[None,:,None,None],projected,kn[:,None])
            dp=2*overlap[...,None]*tangent/norm.clamp_min(1e-12)[None,:,None,None]
            raw=(p.double()+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
            g=dp.double()/(p.double()+EPS)[...,None]
    centered=g-torch.einsum('brt,brtj->brj',alpha,g)[:,:,None]
    weighted=centered*alpha.sqrt()[...,None]
    fisher=torch.einsum('brtj,brtk->rjk',weighted,weighted)/len(h)
    fisher=(fisher+fisher.transpose(-1,-2))/2
    assert torch.isfinite(fisher).all()
    if jacobian:return fisher,alpha,alpha[...,None]*centered
    return fisher

class QueryStep(nn.Module):
    def __init__(self,mode,native=None):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.base=BASE[mode]
        self.native=Head(self.base) if native is None else native
        self.metric=None

    @property
    def head(self):return self.native.head

    def forward(self,h,roles,mask):
        result=self.native(h,roles,mask)
        if self.training and torch.is_grad_enabled() and self.mode!='quantum_adam':
            if self.metric is not None:raise RuntimeError('Unconsumed metric from previous forward')
            self.metric=routing_metric(self,(h,roles,mask))
        return result

class QueryMetricAdamW(torch.optim.AdamW):
    """Keep AdamW moments/body and row-step norm; reshape query directions only."""
    def __init__(self,model,**kwargs):
        super().__init__(model.parameters(),**kwargs)
        self.model=model;self.last_check={}

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Dense accepted callback supplies no closure')
        model=self.model;metric=model.metric
        if metric is None:raise RuntimeError('This exact batch has no query metric')
        q=query_parameter(model);before=q.clone();group=self.param_groups[0]
        assert all(p is not q for g in self.param_groups[1:] for p in g['params'])
        result=super().step()
        retained=before*(1-group['lr']*group['weight_decay'])
        delta=(retained-q).reshape(40,12).double()
        eig,vectors=torch.linalg.eigh(metric)
        scale=metric.diagonal(dim1=-2,dim2=-1).sum(-1)/12
        assert eig.min()>=-1e-7*max(1.,float(scale.max()))
        damping=DAMPING_REL*scale+DAMPING_ABS
        factors=(eig.clamp_min(0)+damping[:,None]).rsqrt()
        shaped=torch.einsum('rjk,rk->rj',vectors,
            factors*torch.einsum('rjk,rj->rk',vectors,delta))
        original_norm=delta.norm(dim=-1,keepdim=True);shaped_norm=shaped.norm(dim=-1,keepdim=True)
        shaped=shaped*(original_norm/shaped_norm.clamp_min(torch.finfo(torch.float64).tiny))
        assert torch.isfinite(shaped).all()
        error=float((shaped.norm(dim=-1)-original_norm[:,0]).abs().max())
        assert error<1e-10
        q.copy_(retained-shaped.reshape_as(q).to(q.dtype))
        stored_error=float(((retained-q).reshape(40,12).double().norm(dim=-1)-original_norm[:,0]).abs().max())
        assert stored_error<5e-7
        self.last_check=dict(step_norm_error=error,minimum_eigenvalue=float(eig.min()),
            maximum_eigenvalue=float(eig.max()),minimum_damping=float(damping.min()),
            maximum_query_step=float(original_norm.max()),stored_fp32_step_norm_error=stored_error)
        model.metric=None
        return result

def make_optimizer(model,cfg):
    kw=dict(lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
    return torch.optim.AdamW(model.parameters(),**kw) if model.mode=='quantum_adam' else QueryMetricAdamW(model,**kw)
