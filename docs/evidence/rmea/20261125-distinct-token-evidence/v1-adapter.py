"""Check only distinct-token/collision-mass statistics; keep thirteen fixed gates."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from shared_query import make_head as make_shared
from moment_head import shared_kind
from plugin import MODES,initialize,make_head,BASE


def independent_statistic(alpha,values,kind):
    left,right=values[...,0::2],values[...,1::2]
    off_diagonal=1-torch.eye(values.shape[1],dtype=values.dtype,device=values.device)
    if kind=='quantum_collision_mass':
        mass=torch.einsum('bri,brj,ij->br',alpha,alpha,off_diagonal)
        mean_product=torch.einsum('bri,brj,bic,bjc->brc',alpha,alpha,left,right)
        return mass[...,None]*mean_product
    return torch.einsum('bri,brj,bic,bjc,ij->brc',alpha,alpha,left,right,off_diagonal)


def fixtures(models,inputs,labels):
    checks={}
    for kind,original in models.items():
        state={n:v.clone() for n,v in original.state_dict().items()}
        m=copy.deepcopy(original).to(inputs[0].device).double().eval()
        x=tuple(t[:2,:6].double() if t.dtype!=torch.bool else t[:2,:6] for t in inputs)
        with torch.no_grad():
            base=make_shared(shared_kind(BASE[kind])).double().to(inputs[0].device)
            base.load_state_dict({n:v for n,v in m.state_dict().items() if n!='moment_decoder'})
            endpoint=float((m(*x)-base(*x)).abs().max());assert endpoint==0
            m.moment_decoder.copy_(torch.linspace(-.3,.4,1280,device=x[0].device,dtype=torch.float64).reshape(40,32))
        alpha,values,mean,z=m.components(*x)
        independent=independent_statistic(alpha,values,kind)
        formula_error=float((m.statistic(alpha,values,mean)-independent).abs().max())
        actual=m(*x);expected=z+(independent*m.moment_decoder).sum(-1)
        assert formula_error<1e-10 and float((actual-expected).abs().max())<1e-10
        got=torch.autograd.grad(actual.square().mean(),tuple(m.parameters()))
        ref=torch.autograd.grad(expected.square().mean(),tuple(m.parameters()))
        grad_error=max(float((a-b).abs().max()) for a,b in zip(got,ref));assert grad_error<1e-8
        query_grad=torch.autograd.grad(F.cross_entropy(m(*x),labels[:2]),m.query_weights)[0]
        idx=int(query_grad.abs().argmax());step=1e-5
        with torch.no_grad():
            w=m.query_weights.reshape(-1);before=w[idx].item()
            w[idx]=before+step;plus=F.cross_entropy(m(*x),labels[:2])
            w[idx]=before-step;minus=F.cross_entropy(m(*x),labels[:2]);w[idx]=before
            fd_error=abs(float((plus-minus)/(2*step))-float(query_grad.reshape(-1)[idx]));assert fd_error<1e-7
            single=tuple(t[:,:1] for t in x)
            single_alpha,single_values,single_mean,_=m.components(*single)
            single_error=float(m.statistic(single_alpha,single_values,single_mean).abs().max());assert single_error==0
            constant=values[:,:1].expand_as(values)
            constant_mean=torch.einsum('brt,btd->brd',alpha,constant)
            constant_expected=(1-alpha.square().sum(-1,keepdim=True))*constant_mean[...,0::2]*constant_mean[...,1::2]
            constant_error=float((m.statistic(alpha,constant,constant_mean)-constant_expected).abs().max());assert constant_error<1e-10
            pad=(F.pad(x[0],(0,0,0,2),value=.8),F.pad(x[1],(0,0,0,2),value=.9),F.pad(x[2],(0,2)))
            assert float((m(*x)-m(*pad)).abs().max())<1e-10
        u=copy.deepcopy(original).to(inputs[0].device);buffers={n:v.clone() for n,v in u.named_buffers()}
        opt=torch.optim.AdamW(u.parameters(),lr=.001,weight_decay=.01)
        old=u.query_weights.detach().clone();loss=F.cross_entropy(u(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in u.parameters())
        assert float(u.moment_decoder.grad.abs().max())>0
        torch.nn.utils.clip_grad_norm_(u.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert not torch.equal(old,u.query_weights) and all(torch.isfinite(p).all() for p in u.parameters())
        assert all(torch.equal(dict(u.named_buffers())[n],v) for n,v in buffers.items())
        assert all(torch.equal(original.state_dict()[n],v) for n,v in state.items())
        count=sum(p.numel() for p in u.parameters());assert count==81924
        checks[kind]=dict(parameters=count,zero_moment_shared_endpoint=endpoint,independent_formula_error=formula_error,
            full_parameter_gradient_error=grad_error,query_finite_difference_error=fd_error,single_token_error=single_error,
            identical_values_error=constant_error,padding=True,real8_update=True,original_models_and_buffers_unchanged=True)
    return dict(status='PASS',checks=checks,scope='Distinct-position deletion and concentration-only control; accepted modules/core reused')


def summarize(rows,logits,labels,cfg):
    assert tuple(rows)==MODES
    for kind,z in logits.items():
        cm=np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40);den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    anchors=cfg['reused_native_rows'];candidate=rows['quantum_distinct'];assert len(anchors)==9
    assert len({v['order_sha256'] for v in list(rows.values())+list(anchors.values())})==1
    assert all(v['class_support']==candidate['class_support'] and v['updates']==1024 for v in anchors.values())
    comparators={k:v for k,v in rows.items() if k!='quantum_distinct'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates={'macro_beats_'+k:candidate['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()};assert len(gates)==13
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_DISTINCT_TOKEN_SCREEN',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
        interpretation='Known off-diagonal quadratic identity, no semantic-binding/coherence/novelty proof. All controls and prior failures retained.')
