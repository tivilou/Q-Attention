"""Focused pair-readout verification and fixed endpoint summaries."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from plugin import MODES,initialize,make_head,pool_pairs,pair_weights
from math_check import numpy_pairs


def direct_formula(model,inputs):
    a,q,u=model.components(*inputs);mask=inputs[2];k=len(q)
    r,s=torch.triu_indices(k,k,offset=1,device=q.device);qr,qs=q[r],q[s]
    c=(qr.conj()*qs).sum(-1);eta=c/c.abs().clamp_min(EPS)
    g=(a[:,r].abs().square()+a[:,s].abs().square())/2
    if model.kind=='quantum_pair_intensity':
        trace_g=(qr.abs().square().sum(-1)+qs.abs().square().sum(-1))/2
        w=g*(1-(eta*c.conj()).real/trace_g)[None,:,None]
    else:w=g-(a[:,r].conj()*eta[None,:,None]*a[:,s]).real
    raw=(w+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
    edge=(alpha*(u[:,r]-u[:,s])).sum(-1)
    incidence=F.one_hot(r,k).to(u.dtype)-F.one_hot(s,k).to(u.dtype)
    return edge@incidence/k+model.head.bias


def fixtures(models,inputs,labels):
    checks={};device=inputs[0].device
    def check(v,tol=1e-9):
        x=float(v);assert np.isfinite(x) and x<tol,(x,tol);return x
    for kind,original in models.items():
        model=copy.deepcopy(original).to(device);double=copy.deepcopy(model).double()
        assert sum(p.numel() for p in model.parameters())==81028
        x=tuple(t[:2,:5].double() if t.dtype!=torch.bool else t[:2,:5] for t in inputs)
        got=double(*x);ref=direct_formula(double,x);formula=check((got-ref).abs().max())
        p=tuple(double.parameters());g=torch.autograd.grad(got.square().mean(),p,retain_graph=True);f=torch.autograd.grad(ref.square().mean(),p)
        gradient=check(max((a-b).abs().max() for a,b in zip(g,f)),1e-7)
        with torch.no_grad():
            amplitude,q,u=double.components(*x)
            if kind=='real12_pair':state=F.normalize(double.head.key(torch.cat(x[:2],-1)),dim=-1)
            else:
                import math
                from baselines.retacred.rmea import circuit
                theta=(math.pi*torch.tanh(double.head.prepare(torch.cat(x[:2],-1)))).reshape(x[0].shape[:2]+(2,3,2));state=circuit(theta)
            expected=np.stack([numpy_pairs(state[i].cpu().numpy(),q.cpu().numpy(),u[i].cpu().numpy(),double.head.bias.cpu().numpy(),x[2][i].cpu().numpy(),kind=='quantum_pair_intensity')[0] for i in range(len(state))])
            numpy_error=check(np.max(abs(expected-got.cpu().numpy())))
            other=pool_pairs(amplitude,q,u,x[2],double.head.bias,kind=='quantum_pair_intensity',block=37)
            check((other-got).abs().max())
            padded=(F.pad(x[0],(0,0,0,2),value=.23),F.pad(x[1],(0,0,0,2),value=.5),F.pad(x[2],(0,2),value=False))
            padding=check((double(*x)-double(*padded)).abs().max())
            one=tuple(t[:1,:1] for t in x);native=(double.head.value(one[0])[:,0]*double.head.decoder).sum(-1)+double.head.bias
            onez=double(*one);check(((onez-onez.mean(-1,keepdim=True))-(native-native.mean(-1,keepdim=True))).abs().max())
            for bad in ((x[0],x[1],torch.zeros_like(x[2])),tuple(t[:,:0] for t in x)):
                try:double(*bad)
                except ValueError:pass
                else:raise AssertionError('Empty input accepted')
            # Actual head's query states give phase gauge and class-permutation checks.
            phase=torch.exp(1j*torch.linspace(-1.1,.7,40,dtype=torch.float64,device=device)) if q.is_complex() else torch.where(torch.arange(40,device=device)%2==0,1.,-1.).double()
            gauge=pool_pairs(amplitude*phase.conj()[None,:,None],q*phase[:,None],u,x[2],double.head.bias,kind=='quantum_pair_intensity')
            check((gauge-got).abs().max())
            perm=torch.arange(39,-1,-1,device=device)
            permz=pool_pairs(amplitude[:,perm],q[perm],u[:,perm],x[2],double.head.bias[perm],kind=='quantum_pair_intensity')
            check((permz-got[:,perm]).abs().max())
            # Coincident and orthogonal queries: positive weights and finite derivatives.
            primitive=torch.eye(4,dtype=q.dtype,device=device);primitive[1]=primitive[0]
            amp=torch.randn(2,4,5,dtype=torch.float64,device=device).to(q.dtype)
            amp[:,1]=amp[:,0];first=torch.tensor([0,0],device=device);second=torch.tensor([1,2],device=device)
            w=pair_weights(amp,primitive,first,second,kind=='quantum_pair_intensity');assert w.min()>=0
            coincident=check(w[:,0].abs().max())
            orthogonal=check((w[:,1]-(amp[:,0].abs().square()+amp[:,2].abs().square())/2).abs().max())
            with torch.enable_grad():
                pq=primitive.clone().requires_grad_();pa=amp.clone().requires_grad_()
                pw=pair_weights(pa,pq,first,second,kind=='quantum_pair_intensity')
                primitive_grad=torch.autograd.grad(pw.sum(),(pa,pq))
                assert all(torch.isfinite(g).all() for g in primitive_grad)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()};loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()) and any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[kind]=dict(parameters=81028,formula_error=formula,gradient_error=gradient,numpy_error=numpy_error,padding_error=padding,coincident_error=coincident,orthogonal_error=orthogonal,primitive_finite_gradients=True,real8_ce=float(loss.detach()),update=True)
    assert all(torch.equal(v,models['quantum_pair_intensity'].state_dict()[n]) for n,v in models['quantum_pair'].state_dict().items())
    return dict(status='PASS',checks=checks,scope='Changed pair effect/readout and cloned8 actual-row updates; unchanged lifecycle reused')


def summarize(rows,logits,labels,cfg):
    predictions={}
    for kind,z in logits.items():
        pred=z.argmax(-1);predictions[kind]=pred;cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        value=z.astype(np.float64);rivals=value.copy();rivals[np.arange(len(labels)),labels]=-np.inf
        margin=value[np.arange(len(labels)),labels]-rivals.max(-1)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),positive_mean_margin=float(margin[labels>0].mean()),positive_rows=int((labels>0).sum()))
    refs=cfg['reused_native_rows'];assert set(refs)=={'quantum_native','real12','softmax12','rbf12'}
    assert {x['order_sha256'] for x in rows.values()}=={x['order_sha256'] for x in refs.values()}
    assert all(x['class_support']==rows['quantum_pair']['class_support'] and x['updates']==512 for x in refs.values())
    q=rows['quantum_pair'];comparators={k:v for k,v in rows.items() if k!='quantum_pair'};comparators.update({'reused/'+k:v for k,v in refs.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()}
    gates['positive_margin_beats_intensity']=q['positive_mean_margin']>rows['quantum_pair_intensity']['positive_mean_margin'];assert len(gates)==7
    correct=predictions['quantum_pair']==labels;paired={}
    for k,pred in predictions.items():
        if k=='quantum_pair':continue
        peer=pred==labels;paired[k]=dict(repair=int((correct&~peer).sum()),damage=int((~correct&peer).sum()),both_correct=int((correct&peer).sum()),both_wrong=int((~correct&~peer).sum()))
    return dict(results=rows,reused_results=refs,gates=gates,paired_outcomes=paired,
        gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_SHARED_PAIR_SCREEN',
        interpretation='Exposed TRAIN512 complete-classifier screen only. Known aligned subtraction/covariance/voting are not newT; all comparisons/mean positive margins preserved.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
