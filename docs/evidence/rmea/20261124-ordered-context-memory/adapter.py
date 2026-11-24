"""Affected prefix/operator checks, real first update and fixed screen gates."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES,BASE,OrderedMemory,initialize,quantum_matrix,quantum_transition,orthogonal_transition,prefix_states

def make_head(mode):return OrderedMemory(mode,Head(BASE[mode]))

def numpy_quantum(angles):
    u=np.eye(8,dtype=np.complex128)
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
            gate=np.diag(np.exp(np.array([-.5j*z,.5j*z])))@np.array([[c,-s],[s,c]])
            u=np.kron(np.kron(np.eye(2**wire),gate),np.eye(2**(2-wire)))@u
        for control,target in ((0,1),(1,2),(2,0)):
            p=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            u=u[p]
    return u

def fixtures(models,inputs,labels):
    device=inputs[0].device;checks={}
    angles=torch.linspace(-1.1,.9,2*3*12,dtype=torch.float64,device=device).reshape(2,3,2,3,2)
    matrices=quantum_matrix(angles)
    checks['native_first_column']=float((matrices[...,0]-circuit(angles)).abs().max())
    num=np.stack([numpy_quantum(a) for a in angles.cpu().numpy().reshape(-1,2,3,2)]).reshape(2,3,8,8)
    checks['numpy_full_operator']=float(np.abs(num-matrices.cpu().numpy()).max())
    zero=quantum_transition(torch.zeros_like(angles))
    checks['zero_transition']=float((zero-torch.eye(8,device=device,dtype=zero.dtype)).abs().max())
    mask=torch.tensor([[True,True,True],[True,True,False]],device=device)
    for base in ('quantum','dephased','real12'):
        x=angles.detach().clone().requires_grad_(True)
        def trans(v):
            if base=='real12':return orthogonal_transition(v.flatten(-3))
            u=quantum_transition(v)
            return u.abs().square() if base=='dephased' else u
        matrix=trans(x);parallel=prefix_states(matrix,mask);serial=prefix_states(matrix,mask,serial=True)
        weight=torch.linspace(-.7,1.3,parallel.numel(),dtype=torch.float64,device=device).reshape(parallel.shape)
        def objective(v):return (v.abs().square()*weight).sum() if v.is_complex() else (v*weight).sum()
        g1=torch.autograd.grad(objective(parallel),x,retain_graph=True)[0]
        g2=torch.autograd.grad(objective(serial),x)[0]
        padding=(parallel[1,2]-parallel[1,1]).abs().max()
        norm=(parallel.sum(-1)-1).abs().max() if base=='dephased' else (parallel.abs().square().sum(-1)-1).abs().max()
        values=dict(prefix_error=float((parallel-serial).abs().max()),gradient_error=float((g1-g2).abs().max()),padding_error=float(padding),norm_error=float(norm))
        assert max(values.values())<1e-11,(base,values)
        if base=='dephased':assert parallel.min()>=0
        checks[base]=values
    assert max(checks[k] for k in ('native_first_column','numpy_full_operator','zero_transition'))<1e-12
    # One-word no-padding unitary memory agrees with original native route.
    clone=copy.deepcopy(models['quantum_seq']).double().to(device)
    one=(inputs[0][:1,:1].double(),inputs[1][:1,:1].double(),inputs[2][:1,:1])
    single=float((clone(*one)-clone.native(*one)).abs().max());assert single<1e-11
    checks['single_word_native_error']=single
    for mode,original in models.items():
        model=copy.deepcopy(original).to(device)
        assert sum(p.numel() for p in model.parameters())==81028
        if mode=='quantum_native':
            with torch.no_grad():assert torch.equal(model(*inputs),model.native(*inputs))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        model.train();loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[mode+'_update']=dict(loss=float(loss.detach()),parameters=81028,finite=True,state_changed=True)
    return dict(status='PASS',checks=checks,scope='Changed operators/prefix/gradients and one cloned real8 update per arm only; no lifecycle matrix.')

def summarize(rows,logits,labels,cfg):
    for mode,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
    q=rows['quantum_seq'];gates={}
    for mode,row in rows.items():
        if mode=='quantum_seq':continue
        gates['macro_beats_'+mode]=q['all40_macro_f1']>row['all40_macro_f1']
        gates['ce_beats_'+mode]=q['ce']<row['ce']
    return dict(results=rows,gates=gates,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_ORDERED_MEMORY_SCREEN',
        interpretation='Exposed TRAIN fixed512; recurrent unitary memory is known, not a new theorem, independence or family impossibility.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
