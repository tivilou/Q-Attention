"""Checks of the changed readout and summaries; stable screen owns execution."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES,Redraw,initialize,quantum_matrix,decision

def make_head(kind):return Redraw(kind,Head('real12' if kind=='classical_record' else 'quantum'))

def numpy_matrix(angles):
    u=np.eye(8,dtype=np.complex128)
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
            gate=np.diag(np.exp(np.array([-.5j*z,.5j*z])))@np.array([[c,-s],[s,c]])
            u=np.kron(np.kron(np.eye(2**wire),gate),np.eye(2**(2-wire)))@u
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            u=u[permutation]
    return u

def numpy_redraw(a,p,u):
    mu=sum(float(a[t])*float(u[t]) for t in range(len(a)))
    keep=[]
    for k in range(p.shape[-1]):
        c=sum(float(a[t])*float(p[t,k])*(float(u[t])-mu) for t in range(len(a)))
        keep.append(c>0)
    return sum(float(a[t])*float(p[t,k])*(float(u[t]) if keep[k] else sum(float(a[j])*float(u[j]) for j in range(len(a)))) for t in range(len(a)) for k in range(p.shape[-1]))

def fixtures(models,inputs,labels):
    device=inputs[0].device;checks={}
    def check(value,tolerance):
        x=float(value);assert np.isfinite(x) and x<tolerance,(x,tolerance);return x
    angles=torch.linspace(-1.1,.8,36,dtype=torch.float64,device=device).reshape(3,2,3,2)
    matrix=quantum_matrix(angles)
    checks['native_first_column']=check((matrix[...,0]-circuit(angles)).abs().max(),1e-12)
    reference=np.stack([numpy_matrix(a) for a in angles.cpu().numpy()])
    checks['numpy_operator']=check(np.max(abs(reference-matrix.cpu().numpy())),1e-12)
    aa=np.array([.4,.6]);pp=np.array([[.7,.3],[.4,.6]]);uu=np.array([1.2,-.8])
    known=numpy_redraw(aa,pp,uu)
    zz,_=decision(*(torch.tensor(q,dtype=torch.float64,device=device) for q in (aa,pp,uu)))
    checks['actual_decision_known_gain']=check(abs(float(zz)-known),1e-12)
    assert known-aa@uu>.1
    for kind,original in models.items():
        model=copy.deepcopy(original).to(device)
        count=sum(p.numel() for p in model.parameters());assert count==(85508 if kind=='classical_record' else 81028)
        double=copy.deepcopy(model).double()
        x=tuple(v[:2,:5].double() if v.dtype!=torch.bool else v[:2,:5] for v in inputs)
        with torch.no_grad():
            a,p,u=double.routing(*x);mu=(a*u).sum(-1)
            check((a.sum(-1)-1).abs().max(),1e-12)
            assert a.min()>=0 and torch.count_nonzero(a.masked_select(~x[2][:,None].expand_as(a)))==0
            if kind=='utility_oracle':keep=(u>mu[...,None]).double();m=keep
            else:
                check((p.sum(-1)-1).abs().max(),1e-12);assert p.min()>=0
                z,c=decision(a,p,u);m=(p*(c>0)[:,:,None]).sum(-1)
                shifted,_=decision(a,p,u+2.3)
                check((shifted-z-2.3).abs().max(),1e-11)
            alpha=a*(1+m-(a*m).sum(-1,keepdim=True))
            assert alpha.min()>=-1e-14
            check((alpha.sum(-1)-1).abs().max(),1e-12)
            formula=check((double(*x)-(alpha*u).sum(-1)-double.head.bias).abs().max(),1e-10)
            if kind!='classical_record':
                head=copy.deepcopy(double.head);head.mode='dephased' if kind=='dephased_record' else 'quantum'
                details=head(*x,details=True)
                check((a-details['alpha']).abs().max(),1e-12)
                if p is not None:check((p[...,0]-details['probability']).abs().max(),1e-12)
            else:
                native=Head('real12').double().to(device);native.head.load_state_dict(double.head.state_dict())
                check((mu+double.head.bias-native(*x)).abs().max(),1e-11)
            if p is not None:
                na,np_,nu=(q.cpu().numpy() for q in (a,p,u));nz=np.empty(mu.shape)
                for bi in range(len(na)):
                    for ri in range(40):nz[bi,ri]=numpy_redraw(na[bi,ri],np_[bi,ri],nu[bi,ri])
                checks[kind+'_numpy_actual_readout']=check(np.max(abs(nz-(double(*x)-double.head.bias).cpu().numpy())),1e-10)
            constant=copy.deepcopy(double);constant.head.value.weight.zero_();constant.head.value.bias.fill_(.1)
            _,_,cu=constant.routing(*x)
            check((constant(*x)-cu[...,0]-constant.head.bias).abs().max(),1e-10)
            for bad in ((x[0],x[1],torch.zeros_like(x[2])),tuple(t[:,:0] for t in x)):
                try:double(*bad)
                except ValueError:pass
                else:raise AssertionError('Empty/allmasked input accepted')
            one=tuple(t[:1,:1] for t in x);a1,p1,u1=double.routing(*one)
            check((double(*one)-u1[...,0]-double.head.bias).abs().max(),1e-11)
            padded=(F.pad(x[0],(0,0,0,2),value=.123),F.pad(x[1],(0,0,0,2),value=1),F.pad(x[2],(0,2),value=False))
            padding=check((double(*x)-double(*padded)).abs().max(),1e-10)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()) and any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[kind]=dict(parameters=count,formula_error=formula,padding_error=padding,real8_loss=float(loss.detach()),finite_gradient=True,changed=True)
    assert all(torch.equal(v,models['utility_oracle'].state_dict()[n]) and torch.equal(v,models['dephased_record'].state_dict()[n]) for n,v in models['quantum_record'].state_dict().items())
    return dict(status='PASS',checks=checks,scope='Only changed measurement/readout and cloned real8 update; stable core/resume/COS reused.')

def summarize(rows,logits,labels,cfg):
    for kind,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    anchors=cfg['reused_native_rows']
    assert set(anchors)=={'quantum_native','real12','softmax12'}
    assert {x['order_sha256'] for x in rows.values()}=={x['order_sha256'] for x in anchors.values()}
    assert all(x['class_support']==rows['quantum_record']['class_support'] and x['updates']==512 for x in anchors.values())
    q=rows['quantum_record'];comparators={k:v for k,v in rows.items() if k!='quantum_record'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()}
    assert len(gates)==6
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_REDRAW_SCREEN',
        interpretation='Known value-of-signal principle; exposed TRAIN fixed512, no newT/independentL3. All four new fits and all six comparisons retained. Native metrics reused from pinned identical-protocol completed study; no historical checkpoint loaded.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
