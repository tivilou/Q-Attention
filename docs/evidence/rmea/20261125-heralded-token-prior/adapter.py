"""Affected prior fixtures and byte-pinned reuse of already completed anchors."""
from pathlib import Path
import copy,hashlib,json,sys,os
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import parameter_schema
from plugin import MODES,BASE,PriorHead,initialize

if not __debug__:raise RuntimeError('Optimized execution invalidates mandatory scientific checks')

def make_head(mode):return PriorHead(mode)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def error(x,tol):
    e=float(x);assert np.isfinite(e) and e<tol,e
    return e

def fixtures(models,inputs,labels):
    checks={};device=inputs[0].device
    short=tuple(v.double() if v.dtype!=torch.bool else v for v in inputs)
    for mode in MODES[:5]:
        original=models[mode]
        schema=parameter_schema(BASE[mode]);state=original.native.state_dict()
        assert set(state)==set(schema) and all(list(v.shape)==schema[n] and v.dtype==torch.float32 for n,v in state.items())
        model=copy.deepcopy(original).to(device);dm=copy.deepcopy(model).double()
        alpha,ll,energy,_=dm.routing(*short);mask=short[2]
        # Independently reconstruct both paths; detach/temperature bugs must fail.
        a=dm.head;x=torch.cat(short[:2],-1)
        if mode in ('quantum_prior','dephased_prior'):
            independent=a(*short,details=True);key=a.prepare(x)
            refll=torch.log(independent['probability']+EPS)
        else:
            key=a.key(x)
            if mode=='real12_prior':
                score=(F.normalize(key,dim=-1)[:,None]*F.normalize(a.query,dim=-1)[None,:,None]).sum(-1).square()
                refll=torch.log(score+EPS)
            else:refll=(key[:,None]*a.query[None,:,None]).sum(-1)/(12 if mode=='rbf12_no_energy' else 12**.5)
        refe=key.square().mean(-1)/2
        if mode=='rbf12_no_energy':raw=torch.exp(refll-refll.amax(-1,keepdim=True))*mask[:,None]
        else:raw=torch.exp(refll)*torch.exp(-refe[:,None])*mask[:,None]
        ref=raw/raw.sum(-1,keepdim=True)
        ae=error((alpha-ref).abs().max(),1e-12)
        w=torch.linspace(-.8,1.2,alpha.numel(),device=device,dtype=torch.float64).reshape_as(alpha)
        ga=torch.autograd.grad((alpha*w).sum(),tuple(dm.parameters()),retain_graph=True,allow_unused=True)
        gr=torch.autograd.grad((ref*w).sum(),tuple(dm.parameters()),retain_graph=True,allow_unused=True)
        ge=max([0.]+[error((a-b).abs().max(),1e-9) for a,b in zip(ga,gr) if a is not None and b is not None])
        assert all((a is None)==(b is None) for a,b in zip(ga,gr))
        if mode in ('quantum_prior','dephased_prior'):
            d=dm.head(*short,details=True)
            zero=ll.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            error((zero-d['alpha']).abs().max(),1e-12)
            error(((zero*d['utilities']).sum(-1)+dm.head.bias-d['logits']).abs().max(),1e-12)
            joint=torch.exp(-energy[:,None])*(d['probability']+EPS)/(1+EPS)*mask[:,None]
            error((alpha-joint/joint.sum(-1,keepdim=True)).abs().max(),1e-12)
            # Exercise the actual FP32 forward with only the prior neutralized.
            zero_model=copy.deepcopy(model)
            def zero_routing(h,roles,mask):
                detail=zero_model.head(h,roles,mask,details=True)
                ll=torch.log(detail['probability']+EPS)
                alpha=ll.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
                return alpha,ll,torch.zeros_like(h[...,0]),detail
            zero_model.routing=zero_routing
            za,_,_,zd=zero_routing(*inputs)
            actual=zero_model(*inputs);expected=(za*zd['utilities']).sum(-1)+zero_model.head.bias
            assert torch.equal(actual,expected),'Native utility-first FP32 reduction changed'
            checks[mode+'_zero_prior_fp32_logits_error']=error((actual-zd['logits']).abs().max(),2e-6)
        with torch.no_grad():
            z=dm(*short);assert torch.isfinite(z).all() and torch.isfinite(alpha).all() and alpha.min()>=0
            error((alpha.sum(-1)-1).abs().max(),1e-12)
            assert torch.count_nonzero(alpha.masked_select(~mask[:,None].expand_as(alpha)))==0
            padded=(F.pad(short[0],(0,0,0,2),value=.123),F.pad(short[1],(0,0,0,2),value=1),F.pad(mask,(0,2),value=False))
            pe=error((z-dm(*padded)).abs().max(),1e-10)
            for bad in (torch.zeros_like(mask),mask.double(),mask[:1]):
                try:dm.routing(short[0],short[1],bad)
                except ValueError:pass
                else:raise AssertionError('Invalid mask accepted')
            if mode=='rbf12_no_energy':
                key=dm.head.key(torch.cat(short[:2],-1));q=dm.head.query
                direct=(-.5*(key[:,None]-q[None,:,None]).square().mean(-1)).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
                decomposed=(ll-energy[:,None]).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
                checks['rbf_decomposition_error']=error((direct-decomposed).abs().max(),1e-12)
        assert sum(p.numel() for p in model.parameters())==81028
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()) and any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[mode]=dict(probability_error=ae,gradient_error=ge,padding_error=pe,parameters=81028,loss=float(loss.detach()),finite=True,state_changed=True)
    for mode in MODES[5:]:
        model=copy.deepcopy(models[mode]).to(device)
        z=model(*inputs)
        if mode!='rbf12':assert torch.equal(z,model.native(*inputs))
        else:
            a=model.head;k=a.key(torch.cat(inputs[:2],-1))
            alpha=(-.5*(k[:,None]-a.query[None,:,None]).square().mean(-1)).masked_fill(~inputs[2][:,None],-torch.inf).softmax(-1)
            ref=(torch.einsum('brt,btd->brd',alpha,a.value(inputs[0]))*a.decoder).sum(-1)+a.bias
            assert torch.equal(z,ref)
        loss=F.cross_entropy(z,labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks[mode]=dict(native_or_previous_formula_parity=True,finite_real8_gradient=True)
    assert all(torch.equal(v,models['dephased_prior'].state_dict()[n]) for n,v in models['quantum_prior'].state_dict().items())
    return dict(status='PASS',checks=checks,scope='New prior algebra/mask/gradients and real8 cloned update only; stable lifecycle reused.')

def summarize(rows,logits,labels,cfg):
    combined=logits;allrows=rows
    for mode,z in combined.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        allrows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
    q=rows['quantum_prior'];gates={'macro_beats_'+m:q['all40_macro_f1']>v['all40_macro_f1'] for m,v in allrows.items() if m!='quantum_prior'}
    new=logits['quantum_prior'].argmax(-1);paired={}
    for mode,z in combined.items():
        if mode=='quantum_prior':continue
        old=z.argmax(-1);repair=(new==labels)&(old!=labels);harm=(new!=labels)&(old==labels)
        paired[mode]=dict(repair=int(repair.sum()),harm=int(harm.sum()),net_correct=int(repair.sum()-harm.sum()),
            repair_by_gold=np.bincount(labels[repair],minlength=40).tolist(),harm_by_gold=np.bincount(labels[harm],minlength=40).tolist())
    assert len(gates)==8
    return dict(results=rows,gates=gates,paired=paired,
        gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_TOKEN_PRIOR_SCREEN',
        interpretation='Exposed TRAIN; all9 conditions fitted in one campaign. Known kernel/prior/postselection algebra is not T. CE secondary.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
