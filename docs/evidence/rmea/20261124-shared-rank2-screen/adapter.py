"""Study factory, affected fixtures and fixed screen interpretation only."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head
from plugin import SharedMeasurement

MODES=('quantum1','quantum2','dephased2','real12_2','softmax12_2')
BASE=dict(zip(MODES,('quantum','quantum','dephased','real12','softmax12')))


def make_head(mode):
    if mode not in MODES:raise ValueError(mode)
    return SharedMeasurement(Head(BASE[mode]),components=1 if mode=='quantum1' else 2)


def initialize(initial,seed):
    models={}
    for mode in MODES:
        parent=initialize_head(BASE[mode],initial,dict(seed=seed,training_dtype='float32'))
        models[mode]=SharedMeasurement(parent,components=1 if mode=='quantum1' else 2,extra_seed=seed+9187)
        assert sum(p.numel() for p in models[mode].parameters())==(81028 if mode=='quantum1' else 81508)
    assert torch.equal(models['softmax12_2'].head.key.weight,models['real12_2'].head.key.weight)
    assert torch.equal(models['softmax12_2'].extra,models['real12_2'].extra)
    assert torch.equal(models['quantum2'].extra,models['dephased2'].extra)
    return models


def independent_mixture(model,inputs):
    """Explicit component products and mass weighting, separate from plugin einsum."""
    h,roles,mask=inputs;a=model.head;x=torch.cat((h,roles),-1)
    outputs=[];masses=[]
    for parameter in (a.effect,model.extra) if model.mode in ('quantum','dephased') else (a.query,model.extra):
        if model.mode in ('quantum','dephased'):
            key=circuit((math.pi*torch.tanh(a.prepare(x))).reshape(h.shape[:2]+(2,3,2)))
            q=circuit(math.pi*torch.tanh(parameter))
            if model.mode=='quantum':p=(q.conj()[None,:,None]*key[:,None]).sum(-1).abs().square()
            else:p=(q.abs().square()[None,:,None]*key.abs().square()[:,None]).sum(-1)
        else:
            key=a.key(x);q=parameter
            if model.mode=='real12':p=(F.normalize(q,dim=-1)[None,:,None]*F.normalize(key,dim=-1)[:,None]).sum(-1).square()
            else:p=((q[None,:,None]*key[:,None]).sum(-1)/math.sqrt(12)).exp()
        raw=(p if model.mode=='softmax12' else p+EPS)*mask[:,None]
        mass=raw.sum(-1);alpha=raw/mass[...,None]
        u=(a.value(h)[:,None]*a.decoder[None,:,None]).sum(-1)
        outputs.append((alpha*u).sum(-1));masses.append(mass)
    total=masses[0]+masses[1]
    return (outputs[0]*masses[0]+outputs[1]*masses[1])/total+a.bias


def fixtures(models,inputs,labels):
    checks={}
    for mode,original in models.items():
        model=copy.deepcopy(original).to(inputs[0].device);before={n:p.detach().clone() for n,p in model.named_parameters()}
        z=model(*inputs);assert z.shape==(len(labels),40) and torch.isfinite(z).all()
        # K1 path retains the native module state and original forward exactly.
        k1=SharedMeasurement(copy.deepcopy(model.native),components=1).to(inputs[0].device)
        checks[mode+'_K1_parity']=float((k1(*inputs)-model.native(*inputs)).abs().max())
        if model.components==2:
            checks[mode+'_mixture']=float((z-independent_mixture(model,inputs)).abs().max())
        h,roles,mask=inputs
        padded=(F.pad(h,(0,0,0,2),value=.123),F.pad(roles,(0,0,0,2),value=1),F.pad(mask,(0,2),value=False))
        checks[mode+'_padding']=float((model(*padded)-z).abs().max())
        loss=F.cross_entropy(z,labels);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(math.isfinite(x) and x<5e-5 for x in checks.values()),checks
    return dict(status='PASS',errors=checks,real_first8_updates=len(models),
        claim='Only new model formulas/K1/padding/finite gradients/one real-label update; no train/resume/COS capability matrix.')


def summarize(rows,logits,labels,cfg):
    for mode,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f1.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
        assert sum(p for p in cm.sum(1))==4000
    q=rows['quantum2'];gates={}
    for mode,row in rows.items():
        if mode=='quantum2':continue
        gates['macro_beats_'+mode]=q['all40_macro_f1']>row['all40_macro_f1']
        gates['ce_beats_'+mode]=q['ce']<row['ce']
    return dict(results=rows,gates=gates,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_K2_SCREEN',
        interpretation='Known low-rank measurement extension; exposed TRAIN partition and fixed512 updates. No independent utility, novelty, new L3/T or all-family impossibility.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
