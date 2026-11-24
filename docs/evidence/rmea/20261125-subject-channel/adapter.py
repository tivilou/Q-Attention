"""Affected channel checks and one fixed all-class endpoint assessment."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit
from plugin import MODES,initialize,make_head,source_density,channel_effect,channel_scores,pool_scores
from math_check import numpy_scores

def direct_scores(states,unitary,source,population):
    rows=[]
    for r in range(len(unitary)):
        out=[]
        for b in range(len(states)):
            transported=(unitary[r]@states[b,source[b]].T).T
            if population:p=transported.abs().square().mean(0)@states[b].abs().square().T
            else:p=(transported.conj()@states[b].T).abs().square().mean(0)
            out.append(p)
        rows.append(torch.stack(out))
    return torch.stack(rows,dim=1)

def fixtures(models,inputs,labels):
    result={}
    def check(x,tol=1e-9):
        x=float(x);assert np.isfinite(x) and x<tol,(x,tol);return x
    for mode,original in models.items():
        model=copy.deepcopy(original).to(inputs[0].device)
        assert sum(p.numel() for p in model.parameters())==81028
        double=copy.deepcopy(model).double();n=min(7,inputs[0].shape[1]);assert n>=2
        x=(inputs[0][:2,:n].double(),inputs[1][:2,:n].double().clone(),torch.ones((2,n),dtype=torch.bool,device=inputs[0].device))
        x[1][...,2]=0;x[1][:,:2,2]=1
        state,u,rho,e,utility=double.components(*x);population=mode=='population_subject_channel'
        p=channel_scores(state,e,population);source=x[2]&(x[1][...,2]==1)
        explicit=direct_scores(state,u,source,population)
        formula=check((p-explicit).abs().max())
        reference=pool_scores(explicit,x[2],utility,double.head.bias);got=double(*x)
        check((got-reference).abs().max())
        params=tuple(double.parameters())
        ga=torch.autograd.grad(got.square().mean(),params,retain_graph=True)
        gb=torch.autograd.grad(reference.square().mean(),params)
        gradient=check(max((a-b).abs().max() for a,b in zip(ga,gb)),1e-7)
        with torch.no_grad():
            restored=make_head(mode).to(device=inputs[0].device,dtype=torch.float64)
            restored.load_state_dict(double.state_dict(),strict=True)
            assert all(torch.equal(v,restored.state_dict()[k]) for k,v in double.state_dict().items())
            factory_error=check((restored(*x)-got).abs().max())
            expected=np.stack([numpy_scores(state[b].detach().cpu().numpy(),u.detach().cpu().numpy(),source[b].cpu().numpy(),population) for b in range(2)])
            numpy_error=check(np.max(abs(expected-p.cpu().numpy())))
            identity=torch.eye(u.shape[-1],dtype=u.dtype,device=u.device)
            unitarity=check((u@u.conj().transpose(-1,-2)-identity).abs().max())
            trace=check((e.diagonal(dim1=-2,dim2=-1).sum(-1)-1).abs().max())
            assert torch.linalg.eigvalsh(e).min()>-1e-9 and p.min()>=0 and p.max()<=1+1e-9
            isotropic=channel_effect(u,identity.expand(2,-1,-1)/len(identity))
            check((isotropic-identity/len(identity)).abs().max())
            padded=(F.pad(x[0],(0,0,0,3),value=.2),F.pad(x[1],(0,0,0,3),value=1),F.pad(x[2],(0,3),value=False))
            padding=check((double(*padded)-got).abs().max())
            one=(x[0][:1,:1],x[1][:1,:1],x[2][:1,:1]);native=(double.head.value(one[0])[:,0]*double.head.decoder).sum(-1)+double.head.bias
            check((double(*one)-native).abs().max())
            phase=torch.exp(1j*torch.linspace(-.8,.9,n,dtype=torch.float64,device=u.device)) if state.is_complex() else torch.where(torch.arange(n,device=u.device)%2==0,1.,-1.).double()
            altered=state*phase[None,:,None];pe=channel_effect(u,source_density(altered,source))
            gauge=check((channel_scores(altered,pe,population)-p).abs().max())
            if mode!='real12_subject_channel':
                first=check((u[...,0]-circuit(math.pi*torch.tanh(double.head.effect))).abs().max())
                vacuum=torch.zeros_like(rho);vacuum[:,0,0]=1
                z=pool_scores(channel_scores(state,channel_effect(u,vacuum),population),x[2],utility,double.head.bias)
                native_head=copy.deepcopy(double.head);native_head.mode='dephased' if population else 'quantum'
                vacuum_parity=check((z-native_head(*x)[0]).abs().max())
            else:first=None;vacuum_parity=None
        missing=x[1].clone();missing[...,2]=0
        for bad in ((x[0],missing,x[2]),(x[0],x[1],torch.zeros_like(x[2])),tuple(t[:0] for t in x)):
            try:double(*bad)
            except ValueError:pass
            else:raise AssertionError('Invalid source/input accepted')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        before={n:v.detach().clone() for n,v in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(v).all() for v in model.parameters()) and any(not torch.equal(before[n],v) for n,v in model.named_parameters())
        result[mode]=dict(parameters=81028,formula_error=formula,gradient_error=gradient,numpy_error=numpy_error,unitarity_error=unitarity,trace_error=trace,padding_error=padding,phase_error=gauge,native_first_column_error=first,vacuum_native_error=vacuum_parity,factory_roundtrip_error=factory_error,real8_ce=float(loss.detach()),update=True)
    assert all(torch.equal(v,models['population_subject_channel'].state_dict()[n]) for n,v in models['quantum_subject_channel'].state_dict().items())
    return dict(status='PASS',checks=result,scope='Changed conditional source/effect; cloned8 real updates. Core/resume/COS unchanged')

def summarize(rows,logits,labels,cfg):
    predictions={}
    for mode,z in logits.items():
        pred=z.argmax(-1);predictions[mode]=pred
        cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40);den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    refs=cfg['reused_native_rows'];assert set(refs)=={'quantum_native','real12','softmax12','rbf12'}
    q=rows['quantum_subject_channel'];assert {v['order_sha256'] for v in rows.values()}=={v['order_sha256'] for v in refs.values()}
    assert all(v['class_support']==q['class_support'] and v['updates']==512 for v in refs.values())
    others={k:v for k,v in rows.items() if k!='quantum_subject_channel'};others.update({'reused/'+k:v for k,v in refs.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in others.items()};assert len(gates)==6
    correct=predictions['quantum_subject_channel']==labels;paired={}
    for mode,pred in predictions.items():
        if mode=='quantum_subject_channel':continue
        peer=pred==labels;paired[mode]=dict(repair=int((correct&~peer).sum()),damage=int((~correct&peer).sum()),both_correct=int((correct&peer).sum()),both_wrong=int((~correct&~peer).sum()))
    return dict(results=rows,reused_results=refs,gates=gates,paired_outcomes=paired,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_SUBJECT_CHANNEL_SCREEN',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',interpretation='Known conjugated density as input-conditioned relation effect; exposed TRAIN pilot, not newT or a causal attribution')
