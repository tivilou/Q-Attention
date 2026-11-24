"""Affected combination checks and fixed all-control screen gate."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES,BRANCH,AnchoredMeasurement,initialize,attention

def make_head(mode):return AnchoredMeasurement(mode)

def fixtures(models,inputs,labels):
    checks={};device=inputs[0].device
    for mode,original in models.items():
        model=copy.deepcopy(original).to(device)
        count=sum(p.numel() for p in model.parameters());assert count==(93857 if mode in BRANCH else 81028)
        with torch.no_grad():assert torch.equal(model(*inputs),model.native(*inputs))
        if mode in BRANCH:
            m=copy.deepcopy(model).double();x=tuple(v.double() if v.dtype!=torch.bool else v for v in inputs)
            expected=Head(BRANCH[mode]).double().to(device)
            state=expected.head.state_dict()
            for n in ('value.weight','value.bias','decoder','bias'):state[n]=m.head.state_dict()[n]
            state.update(m.route.state_dict());expected.head.load_state_dict(state,strict=True)
            error=0.
            with torch.no_grad():
                u=torch.einsum('btd,rd->brt',m.head.value(x[0]),m.head.decoder)
                w=x[2][:,None].double()/x[2].sum(-1)[:,None,None]
                uniform=(w*u).sum(-1)+m.head.bias
                alpha=attention(m.route,BRANCH[mode],*x)
                assert (alpha.sum(-1)-1).abs().max()<1e-12
                shift=torch.linspace(-2,2,40,device=device,dtype=torch.float64)[None,:,None]
                origin=float((((alpha-w)*(u+shift)).sum(-1)-((alpha-w)*u).sum(-1)).abs().max())
                assert origin<1e-11
                for gate in (-1.,.5,1.):
                    m.gate.fill_(gate)
                    reference=m.native(*x)+gate*(expected(*x)-uniform)
                    gate_error=float((m(*x)-reference).abs().max())
                    assert np.isfinite(gate_error) and gate_error<1e-10
                    error=max(error,gate_error)
                padded=(F.pad(x[0],(0,0,0,2),value=.234),F.pad(x[1],(0,0,0,2),value=1),F.pad(x[2],(0,2),value=False))
                padding=float((m(*x)-m(*padded)).abs().max())
                assert np.isfinite(padding) and padding<1e-10
            checks[mode+'_formula']=dict(native_difference_error=error,padding_error=padding,translation_error=origin)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if mode in BRANCH:
            assert float(model.gate.grad)!=0
            assert all(torch.count_nonzero(p.grad)==0 for p in model.route.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[mode+'_update']=dict(parameters=count,loss=float(loss.detach()),finite=True,state_changed=True,
            gate_after=float(model.gate.detach()) if mode in BRANCH else None)
    q=models['real_plus_quantum'].route.state_dict();d=models['real_plus_dephased'].route.state_dict()
    assert set(q)==set(d) and all(torch.equal(q[n],d[n]) for n in q)
    assert all(torch.equal(models[m].native.state_dict()[n],models['real_native'].native.state_dict()[n])
               for m in BRANCH for n in models['real_native'].native.state_dict())
    return dict(status='PASS',checks=checks,quantum_dephased_initial_exact=True,hybrid_native_initial_exact=True,
        scope='New centered combination/reference/padding and cloned first8 update; no lifecycle matrix.')

def summarize(rows,logits,labels,cfg):
    for mode,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
    q=rows['real_plus_quantum'];gates={}
    for mode,row in rows.items():
        if mode=='real_plus_quantum':continue
        gates['macro_beats_'+mode]=q['all40_macro_f1']>row['all40_macro_f1']
        gates['ce_beats_'+mode]=q['ce']<row['ce']
    return dict(results=rows,gates=gates,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_CLASSICAL_ANCHOR_SCREEN',
        interpretation='Exposed fixedTRAIN/512; residual/differential principle known, not new theory or independentL3.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
