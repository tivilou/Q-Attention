"""Affected channel mathematics and compact reporting; no execution changes."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from plugin import MODES,layer,layer_matrix,initialize,make_head


def numpy_layer(angles):
    u=np.eye(8,dtype=np.complex128)
    for wire in range(3):
        y,z=angles[wire];c,s=np.cos(y/2),np.sin(y/2)
        gate=np.diag(np.exp(np.array([-.5j*z,.5j*z])))@np.array([[c,-s],[s,c]])
        u=np.kron(np.kron(np.eye(2**wire),gate),np.eye(2**(2-wire)))@u
    for control,target in ((0,1),(1,2),(2,0)):
        p=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
        u=u[p]
    return u


def density_formula(model,inputs):
    h,roles,mask=inputs;a=model.head
    if model.kind=='classical_gate':
        key=a.key(torch.cat((h,roles),-1));k=F.normalize(key,dim=-1);q=F.normalize(a.query,dim=-1)
        rho=k[..., :,None]*k[...,None,:]
        lam=model.gate(torch.tanh(key)).sigmoid().squeeze(-1)
        mixed=(1-lam[...,None,None])*rho+lam[...,None,None]*torch.diag_embed(k.square())
        p=torch.einsum('ri,btij,rj->brt',q,mixed,q)
    else:
        f=torch.tanh(a.prepare(torch.cat((h,roles),-1)));theta=(math.pi*f).reshape(h.shape[:2]+(2,3,2))
        u1=layer_matrix(theta[...,0,:,:]);u2=layer_matrix(theta[...,1,:,:]);psi1=u1[...,0]
        rho=psi1[..., :,None]*psi1.conj()[...,None,:]
        lam=torch.ones(h.shape[:2],dtype=h.dtype,device=h.device) if model.kind=='quantum_midfixed' else model.gate(f).sigmoid().squeeze(-1)
        if model.kind=='quantum_outgate':rho=u2@rho@u2.conj().transpose(-1,-2)
        mixed=(1-lam[...,None,None])*rho+lam[...,None,None]*torch.diag_embed(rho.diagonal(dim1=-2,dim2=-1))
        if model.kind!='quantum_outgate':mixed=u2@mixed@u2.conj().transpose(-1,-2)
        phi=circuit(math.pi*torch.tanh(a.effect))
        p=torch.einsum('ri,btij,rj->brt',phi.conj(),mixed,phi).real
    raw=(p+EPS)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
    u=torch.einsum('btd,rd->brt',a.value(h),a.decoder)
    return (alpha*u).sum(-1)+a.bias


def fixtures(models,inputs,labels):
    checks={};device=inputs[0].device
    def check(v,tol=1e-10):
        x=float(v);assert np.isfinite(x) and x<tol,(x,tol);return x
    angles=torch.linspace(-1.2,1.8,36,dtype=torch.float64,device=device).reshape(3,2,3,2)
    a,b=layer_matrix(angles[:,0]),layer_matrix(angles[:,1])
    checks['native_composition']=check(((b@a)[...,0]-circuit(angles)).abs().max())
    checks['independent_numpy_layer']=check(max(np.max(abs(numpy_layer(x)-y)) for x,y in zip(angles[:,1].cpu().numpy(),b.cpu().numpy())))
    for kind,original in models.items():
        model=copy.deepcopy(original).to(device);double=copy.deepcopy(model).double()
        count=sum(p.numel() for p in model.parameters());assert count==(81028 if kind=='quantum_midfixed' else 81041)
        x=tuple(t[:2,:5].double() if t.dtype!=torch.bool else t[:2,:5] for t in inputs)
        # Nonzero, input-dependent gate fixture, not only its zero-weight initialization.
        if kind!='quantum_midfixed':
            with torch.no_grad():double.gate.weight.copy_(torch.linspace(-.7,.8,12,device=device)[None]);double.gate.bias.fill_(.2)
        got=double(*x);expected=density_formula(double,x)
        formula=check((got-expected).abs().max())
        p=tuple(double.parameters());g=torch.autograd.grad(got.square().mean(),p,retain_graph=True);f=torch.autograd.grad(expected.square().mean(),p)
        gradient=check(max((a-b).abs().max() for a,b in zip(g,f)),1e-8)
        with torch.no_grad():
            if kind!='classical_gate':
                a=double.head;theta=math.pi*torch.tanh(a.prepare(torch.cat(x[:2],-1))).reshape(x[0].shape[:2]+(2,3,2))
                phi=circuit(math.pi*torch.tanh(a.effect)).cpu().numpy();coh,endpoint,lam=double.components(*x)
                np_end=np.empty(endpoint.shape)
                for bi in range(len(np_end)):
                    for ti in range(np_end.shape[-1]):
                        th=theta[bi,ti].cpu().numpy();u1=numpy_layer(th[0]);u2=numpy_layer(th[1]);psi=u1[:,0]
                        rho=np.diag(abs(psi)**2)
                        if kind=='quantum_outgate':rho=np.diag(abs(u2@psi)**2)
                        else:rho=u2@rho@u2.conj().T
                        np_end[bi,:,ti]=np.einsum('ri,ij,rj->r',phi.conj(),rho,phi).real
                check(np.max(abs(np_end-endpoint.cpu().numpy())))
                if kind!='quantum_midfixed':
                    old=double.gate.bias.clone();double.gate.weight.zero_();double.gate.bias.fill_(-1000)
                    native=a(*x)[0];check((double(*x)-native).abs().max())
                    double.gate.bias.copy_(old)
            padded=(F.pad(x[0],(0,0,0,2),value=.7),F.pad(x[1],(0,0,0,2),value=.3),F.pad(x[2],(0,2),value=False))
            padding=check((double(*x)-double(*padded)).abs().max())
            one=tuple(t[:1,:1] for t in x);utility=(double.head.value(one[0])[:,0]*double.head.decoder).sum(-1)+double.head.bias
            check((double(*one)-utility).abs().max())
            for bad in ((x[0],x[1],torch.zeros_like(x[2])),tuple(t[:,:0] for t in x)):
                try:double(*bad)
                except ValueError:pass
                else:raise AssertionError('Empty input accepted')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()};loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if kind!='quantum_midfixed':assert model.gate.weight.grad.abs().max()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()) and any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[kind]=dict(parameters=count,formula_error=formula,gradient_error=gradient,padding_error=padding,real8_ce=float(loss.detach()),update=True)
    return dict(status='PASS',checks=checks,scope='Changed layer/channel and cloned8-row update only; stable lifecycle reused')


def summarize(rows,logits,labels,cfg):
    for kind,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    anchors=cfg['reused_native_rows'];assert set(anchors)=={'quantum_native','real12','softmax12','rbf12'}
    assert {x['order_sha256'] for x in rows.values()}=={x['order_sha256'] for x in anchors.values()}
    assert all(x['class_support']==rows['quantum_midgate']['class_support'] and x['updates']==512 for x in anchors.values())
    q=rows['quantum_midgate'];comparators={k:v for k,v in rows.items() if k!='quantum_midgate'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()};assert len(gates)==7
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_INTERLAYER_SCREEN',
        interpretation='Exposed TRAIN512 pilot; known CP/noncommutativity tools, no independent L3/T. All four fits/seven frozen Macro comparisons retained.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
