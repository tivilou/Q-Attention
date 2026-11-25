"""Check only the new query/moment combination; retain every fixed outcome."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from shared_query import make_head as make_shared
from plugin import MODES,initialize,make_head,shared_kind


def independent_statistic(alpha,values,kind):
    left,right=values[...,0::2],values[...,1::2]
    if kind.endswith('mean_product'):
        return torch.einsum('bri,brj,bic,bjc->brc',alpha,alpha,left,right)
    differences=(left[:,:,None]-left[:,None])*(right[:,:,None]-right[:,None])
    return .5*torch.einsum('bri,brj,bijc->brc',alpha,alpha,differences)


def fixtures(models, inputs, labels):
    checks={}
    for kind,original in models.items():
        original_state={n:v.clone() for n,v in original.state_dict().items()}
        m=copy.deepcopy(original).to(inputs[0].device).double().eval()
        x=tuple(t[:2,:6].double() if t.dtype!=torch.bool else t[:2,:6] for t in inputs)
        with torch.no_grad():
            base=make_shared(shared_kind(kind)).double().to(inputs[0].device)
            base.load_state_dict({n:v for n,v in m.state_dict().items() if n!='moment_decoder'})
            endpoint=float((m(*x)-base(*x)).abs().max());assert endpoint==0.
            m.moment_decoder.copy_(torch.linspace(-.3,.4,1280,device=x[0].device,dtype=torch.float64).reshape(40,32))
        alpha,values,mean,z=m.components(*x)
        independent=independent_statistic(alpha,values,kind)
        formula_error=float((m.statistic(alpha,values,mean)-independent).abs().max())
        expected=z+(independent*m.moment_decoder).sum(-1)
        actual=m(*x);assert formula_error<1e-10 and float((actual-expected).abs().max())<1e-10
        got=torch.autograd.grad(actual.square().mean(),tuple(m.parameters()))
        ref=torch.autograd.grad(expected.square().mean(),tuple(m.parameters()))
        grad_error=max(float((a-b).abs().max()) for a,b in zip(got,ref));assert grad_error<1e-8
        loss=F.cross_entropy(m(*x),labels[:2]);query_grad=torch.autograd.grad(loss,m.query_weights)[0]
        idx=int(query_grad.abs().argmax());step=1e-5
        with torch.no_grad():
            w=m.query_weights.reshape(-1);before=w[idx].item()
            w[idx]=before+step;plus=F.cross_entropy(m(*x),labels[:2])
            w[idx]=before-step;minus=F.cross_entropy(m(*x),labels[:2]);w[idx]=before
            fd_error=abs(float((plus-minus)/(2*step))-float(query_grad.reshape(-1)[idx]));assert fd_error<1e-7
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
        assert all(torch.equal(original.state_dict()[n],v) for n,v in original_state.items())
        count=sum(p.numel() for p in u.parameters());assert count==81924
        checks[kind]=dict(parameters=count,zero_moment_shared_endpoint=endpoint,independent_moment_formula_error=formula_error,
            full_parameter_gradient_error=grad_error,nonzero_moment_query_finite_difference_error=fd_error,
            padding=True,real8_update=True,original_models_and_buffers_unchanged=True)
    return dict(status='PASS',checks=checks,scope='New shared-query/moment interaction only; accepted query and core coverage reused')


def summarize(rows,logits,labels,cfg):
    assert tuple(rows)==MODES
    for kind,z in logits.items():
        cm=np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40);den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    anchors=cfg['reused_native_rows'];candidate=rows['quantum_shared_covariance']
    assert set(anchors)=={'shared_quantum','native_quantum_covariance','native_gaussian_covariance','shared_gaussian'}
    assert len({v['order_sha256'] for v in list(rows.values())+list(anchors.values())})==1
    assert all(v['class_support']==candidate['class_support'] and v['updates']==1024 for v in anchors.values())
    comparators={k:v for k,v in rows.items() if k!='quantum_shared_covariance'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates={'macro_beats_'+k:candidate['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()}
    assert len(gates)==8
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_SEMANTIC_CONNECTED_SCREEN',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
        interpretation='Known sharing and second-order pooling, unconfirmed joint hypothesis. No causal/novelty/quantum-advantage claim. Preserve all outcomes and closed profiles; no rank/tokenization/pair/channel/scale/threshold/LR/seed/cohort/budget/control rescue.')
