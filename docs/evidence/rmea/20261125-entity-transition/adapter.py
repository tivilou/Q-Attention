"""Only new transition checks and existing compact assessment conventions."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES,initialize,make_head,probabilities,pool
from math_check import pure_reference
from baselines.retacred.rmea import circuit

def amplitude_reference(states,transition,relation,roles,mask,population,drop):
    out=[]
    for b in range(len(states)):
        s=states[b,mask[b]&(roles[b,:,2]==1)]
        o=states[b,mask[b]&(roles[b,:,3]==1)]
        if drop:s=torch.eye(states.shape[-1],dtype=states.dtype,device=states.device)[:1]
        if population:
            a=torch.einsum('tij,sj->tsi',transition[b].abs().square(),s.abs().square())
            c=torch.einsum('rij,oj->roi',relation.abs().square(),o.abs().square())
            p=torch.einsum('roi,tsi->rtos',c,a).mean((-1,-2))
        else:
            a=torch.einsum('tij,sj->tsi',transition[b],s)
            c=torch.einsum('rij,oj->roi',relation,o)
            p=torch.einsum('roi,tsi->rtos',c.conj(),a).abs().square().mean((-1,-2))
        out.append(p)
    return torch.stack(out)

def fixtures(models,inputs,labels):
    result={}
    def check(x,tol=1e-9):
        x=float(x);assert np.isfinite(x) and x<tol,(x,tol);return x
    for mode,base in models.items():
        model=copy.deepcopy(base).to(inputs[0].device);assert sum(p.numel() for p in model.parameters())==81028
        double=copy.deepcopy(model).double();x=tuple(t[:1].double() if t.dtype!=torch.bool else t[:1] for t in inputs)
        states,t,v,s,o,u=double.components(*x);pop=mode=='population_transition';drop=mode=='quantum_drop_subject'
        p=probabilities(t,v,s,o,pop);ref=amplitude_reference(states,t,v,x[1],x[2],pop,drop)
        formula=check((p-ref).abs().max())
        expected=pool(ref,x[2],u,double.head.bias);got=double(*x)
        check((expected-got).abs().max())
        params=tuple(double.parameters());ga=torch.autograd.grad(got.square().mean(),params,retain_graph=True)
        gb=torch.autograd.grad(expected.square().mean(),params)
        gradient=check(max((a-b).abs().max() for a,b in zip(ga,gb)),1e-7)
        with torch.no_grad():
            sm=(x[2][0]&(x[1][0,:,2]==1)).cpu().numpy();om=(x[2][0]&(x[1][0,:,3]==1)).cpu().numpy()
            numeric=pure_reference(states[0].cpu().numpy(),t[0,:3].cpu().numpy(),v[:2].cpu().numpy(),sm,om,pop,drop)
            numpy_error=check(np.max(abs(numeric-p[0,:2,:3].cpu().numpy())))
            vacuum=torch.zeros_like(s);vacuum[:,0,0]=1
            vac=probabilities(t,v,vacuum,vacuum,pop)
            if mode=='real12_transition':
                native_states=states;native_queries=v[...,0]
            else:
                angles=(math.pi*torch.tanh(double.head.prepare(torch.cat((x[0],x[1]),-1)))).reshape(x[0].shape[:2]+(2,3,2))
                native_states=circuit(angles);native_queries=circuit(math.pi*torch.tanh(double.head.effect))
            check((states-native_states).abs().max())
            native=torch.einsum('ri,bti->brt',native_queries.abs().square(),native_states.abs().square()) if pop else torch.einsum('ri,bti->brt',native_queries.conj(),native_states).abs().square()
            vacuum_error=check((vac-native).abs().max())
            restored=make_head(mode).to(device=inputs[0].device,dtype=torch.float64);restored.load_state_dict(double.state_dict(),strict=True)
            factory=check((restored(*x)-got).abs().max())
            padded=(F.pad(x[0],(0,0,0,3),value=.2),F.pad(x[1],(0,0,0,3),value=1),F.pad(x[2],(0,3),value=False))
            padding=check((double(*padded)-got).abs().max())
            one_roles=x[1][:,:1].clone();one_roles[...,2:]=1
            one=(x[0][:,:1],one_roles,torch.ones_like(x[2][:,:1]))
            one_expected=torch.einsum('btd,rd->br',double.head.value(one[0]),double.head.decoder)+double.head.bias
            check((double(*one)-one_expected).abs().max())
            assert p.min()>-1e-9 and p.max()<1+1e-9
            assert all(torch.linalg.eigvalsh(d).min()>-1e-9 for d in (s,o))
            assert all((d.diagonal(dim1=-2,dim2=-1).sum(-1)-1).abs().max()<1e-9 for d in (s,o))
        bad_roles=x[1].clone();bad_roles[...,3]=0
        for bad in ((x[0],bad_roles,x[2]),(x[0],x[1],torch.zeros_like(x[2])),tuple(a[:0] for a in x)):
            try:double(*bad)
            except ValueError:pass
            else:raise AssertionError('Invalid input accepted')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        before={n:a.detach().clone() for n,a in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()) and any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        result[mode]=dict(parameters=81028,formula_error=formula,gradient_error=gradient,numpy_error=numpy_error,vacuum_parity_error=vacuum_error,padding_error=padding,factory_roundtrip_error=factory,real8_ce=float(loss.detach()),update=True)
    return dict(status='PASS',checks=result,scope='New transition rule and actual cloned8 updates only; stable runner/recovery unchanged')

def summarize(rows,logits,labels,cfg):
    preds={}
    for mode,z in logits.items():
        pred=z.argmax(-1);preds[mode]=pred;cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    refs=cfg['reused_native_rows'];assert set(refs)=={'quantum_native','real12','softmax12','rbf12'}
    q=rows['quantum_transition'];assert {r['order_sha256'] for r in rows.values()}=={r['order_sha256'] for r in refs.values()}
    assert all(r['class_support']==q['class_support'] and r['updates']==512 for r in refs.values())
    others={k:v for k,v in rows.items() if k!='quantum_transition'};others.update({'reused/'+k:v for k,v in refs.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in others.items()};assert len(gates)==7
    paired={};correct=preds['quantum_transition']==labels
    for mode,pred in preds.items():
        if mode=='quantum_transition':continue
        peer=pred==labels;paired[mode]=dict(repair=int((correct&~peer).sum()),damage=int((~correct&peer).sum()),both_correct=int((correct&peer).sum()),both_wrong=int((~correct&~peer).sum()))
    return dict(results=rows,reused_results=refs,gates=gates,paired_outcomes=paired,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_ENTITY_TRANSITION_SCREEN',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',interpretation='Known transition-probability tools; exposed pilot, no causal or theory acceptance')
