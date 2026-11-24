"""Actual normalized-route query chain; pure diagnostic, no optimizer."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS

def parameter(model,mode):return model.head.effect if mode=='quantum' else model.head.query

@torch.no_grad()
def query_differential(model,mode):
    if mode!='quantum':return None
    effect=model.head.effect;tanh=torch.tanh(effect).double();angles=(math.pi*torch.tanh(effect)).double()
    shifted=angles.reshape(40,1,12)+math.pi*torch.eye(12,device=effect.device,dtype=torch.float64)[None]
    return circuit(shifted.reshape(40,12,2,3,2))*(math.pi/2*(1-tanh.square())).reshape(40,12,1)

@torch.no_grad()
def query_gradient(model,mode,inputs,labels,dq=None):
    h,roles,mask=inputs;a=model.head;x=torch.cat((h,roles),-1)
    z=model(*inputs);u=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
    if mode=='quantum':
        key=circuit((math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(2,3,2)))
        q=circuit(math.pi*torch.tanh(a.effect))
        overlap=torch.einsum('rd,btd->brt',q.conj(),key)
        derivative=torch.einsum('rjd,btd->brtj',(query_differential(model,mode) if dq is None else dq).conj(),key.to(torch.complex128))
        p=overlap.abs().square();dp=2*(overlap.to(torch.complex128).conj()[...,None]*derivative).real
    else:
        key=F.normalize(a.key(x),dim=-1);q=F.normalize(a.query,dim=-1)
        overlap=torch.einsum('rd,btd->brt',q,key);p=overlap.square()
        norm=a.query.norm(dim=-1);projected=key.double()[:,None]-overlap.double()[...,None]*q.double()[None,:,None]
        tangent=torch.where((norm>=1e-12)[None,:,None,None],projected,key.double()[:,None])
        dp=2*overlap.double()[...,None]*tangent/norm.double().clamp_min(1e-12)[None,:,None,None]
    raw=(p.double()+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
    center=u.double()-(alpha*u.double()).sum(-1,keepdim=True)
    dz=(alpha[...,None]*center[...,None]*dp.double()/(p.double()+EPS)[...,None]).sum(-2)
    coefficient=z.double().softmax(-1)-F.one_hot(labels,40)
    g=(coefficient[...,None]*dz).reshape(len(h),480)
    assert torch.isfinite(g).all()
    return g,z,dict(probability=p,alpha=alpha,utilities=u,dp=dp,dz=dz)

def fixtures(model,mode,inputs,labels):
    import copy
    out={}
    for dtype in (torch.float64,torch.float32):
        m=copy.deepcopy(model).to(dtype=dtype)
        x=tuple(v[:3].to(dtype=dtype) if v.dtype!=torch.bool else v[:3] for v in inputs);y=labels[:3]
        actual,z,detail=query_gradient(m,mode,x,y)
        losses=F.cross_entropy(m(*x),y,reduction='none');errors=[];absolute_errors=[]
        for i in range(len(y)):
            expected=torch.autograd.grad(losses[i],parameter(m,mode),retain_graph=True)[0].reshape(480).double()
            errors.append(float(((expected-actual[i]).abs()/(1+expected.abs())).max()))
            absolute_errors.append(float((expected-actual[i]).abs().max()))
        error=max(absolute_errors if dtype==torch.float64 else errors)
        assert error<(1e-8 if dtype==torch.float64 else 2e-4),(mode,str(dtype),error)
        # A relation-constant value translation must cancel from the route derivative.
        alpha=detail['alpha'];u=detail['utilities'].double();offset=torch.linspace(-2,2,40,device=u.device)[None,:,None]
        centered=(u-offset)-(alpha*(u-offset)).sum(-1,keepdim=True)
        shifted=(alpha[...,None]*centered[...,None]*detail['dp'].double()/(detail['probability'].double()+EPS)[...,None]).sum(-2)
        origin_error=float((shifted-detail['dz']).abs().max());assert origin_error<1e-8
        readout=(alpha*u).sum(-1)+m.head.bias.double()
        readout_error=float((readout-z.double()).abs().max());assert readout_error<(1e-10 if dtype==torch.float64 else 1e-4)
        out[str(dtype)]=dict(per_example_all480_autograd_scaled_errors=errors,per_example_all480_autograd_absolute_errors=absolute_errors,constant_value_translation_error=origin_error,readout_error=readout_error)
    return out

def aggregate(g,y,correct,labels):
    norm2=np.einsum('ij,ij->i',g,g);mean=g.mean(0);support=np.bincount(y,minlength=40)
    means=np.stack([g[y==c].mean(0) if support[c] else np.zeros(480) for c in range(40)])
    balanced=means.mean(0);length=np.linalg.norm(means,axis=-1);den=length[:,None]*length[None,:]
    cosine=np.divide(means@means.T,den,out=np.zeros((40,40)),where=den!=0)
    def describe(mask):
        rows=g[mask];n=len(rows)
        if not n:return dict(rows=0)
        v=rows.mean(0);second=float(norm2[mask].mean());norm=float(v@v)
        coherence=norm/second if second else 0.
        assert -1e-12<=coherence<=1+1e-10
        return dict(rows=n,mean_gradient_norm=float(np.sqrt(norm2[mask]).mean()),
            mean_squared_gradient_norm=second,squared_mean_vector_norm=norm,coherence=coherence)
    def alignment(direction):
        dots=g@direction
        return dict(positive=int((dots>0).sum()),negative=int((dots<0).sum()),zero=int((dots==0).sum()),
            quantile_probabilities=[0,.01,.1,.5,.9,.99,1],quantiles=np.quantile(dots,[0,.01,.1,.5,.9,.99,1]).tolist())
    rows=[]
    for c in range(40):
        row=describe(y==c);row.update(label=labels[c],mean_dot_sample_weighted=float(means[c]@mean),mean_dot_equal_class=float(means[c]@balanced));rows.append(row)
    reconstruction=(means*support[:,None]).sum(0)/len(y)
    assert np.max(np.abs(reconstruction-mean))<1e-10
    return dict(all=describe(np.ones(len(y),bool)),correct=describe(correct),wrong=describe(~correct),per_class=rows,
        class_mean_cosines=cosine.tolist(),sample_weighted_mean_alignment=alignment(mean),equal_class_mean_alignment=alignment(balanced),
        mean_reconstruction_error=float(np.max(np.abs(reconstruction-mean))),
        class_means_opposed_to_sample_weighted=int(((means@mean)<0).sum()),
        class_means_opposed_to_equal_class=int(((means@balanced)<0).sum()))
