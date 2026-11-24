"""One hypothesis factory, affected derivative/update fixtures and fixed gates."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from baselines.retacred.rmea_matched_routers import initialize_head
from plugin import MODES,BASE,QueryStep,routing_metric,query_parameter,make_optimizer

def make_head(mode):return QueryStep(mode)

def initialize(initial,seed):
    return {mode:QueryStep(mode,initialize_head(BASE[mode],initial,dict(seed=seed,training_dtype='float32'))) for mode in MODES}

def alpha_from_query(model,inputs,query):
    a=model.head;h,roles,mask=inputs
    if model.base in ('quantum','dephased'):
        return torch.func.functional_call(a,{'effect':query},(h,roles,mask),{'details':True})['alpha']
    key=a.key(torch.cat((h,roles),-1))
    if model.base=='softmax12':
        score=torch.einsum('btd,rd->brt',key,query)/math.sqrt(12)
        return score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
    p=torch.einsum('btd,rd->brt',F.normalize(key,dim=-1),F.normalize(query,dim=-1)).square()
    raw=(p+EPS)*mask[:,None];return raw/raw.sum(-1,keepdim=True)

def fixtures(models,inputs,labels):
    checks={};optcfg=dict(lr=.001,betas=[.9,.999],optimizer_eps=1e-8,weight_decay=.01)
    for mode,original in models.items():
        model=copy.deepcopy(original).to(inputs[0].device)
        with torch.no_grad():
            z=model(*inputs);native=model.native(*inputs)
            assert torch.equal(z,native)
        if mode!='quantum_adam':
            # All40 rows/all12 query coordinates against the original autograd.
            m=copy.deepcopy(model).double().eval()
            short=tuple(x[:1,:3].to(dtype=torch.float64) if x.dtype!=torch.bool else x[:1,:3] for x in inputs)
            fm,alpha,jac=routing_metric(m,short,jacobian=True);query=query_parameter(m)
            reference=alpha_from_query(m,short,query)
            ae=float((alpha-reference).abs().max());je=0.
            for j in range(12):
                direction=torch.zeros((40,12),dtype=torch.float64,device=query.device);direction[:,j]=1
                _,derivative=torch.autograd.functional.jvp(lambda q:alpha_from_query(m,short,q),query,direction.reshape_as(query),strict=True)
                je=max(je,float((derivative-jac[...,j]).abs().max()))
            padded=(F.pad(short[0],(0,0,0,2),value=.123),F.pad(short[1],(0,0,0,2),value=1),F.pad(short[2],(0,2),value=False))
            pe=float((routing_metric(m,padded)-fm).abs().max())
            # Sum_t d alpha = 0, including zero/masked token rows.
            se=float(jac.sum(-2).abs().max());assert max(ae,je,pe,se)<1e-7,(mode,ae,je,pe,se)
            checks[mode+'_derivative']=dict(alpha_error=ae,query_jvp_error=je,padding_metric_error=pe,normalization_tangent_error=se)
            if model.base=='real12':
                with torch.no_grad():
                    m.head.query[0].zero_();m.head.query[1].zero_();m.head.query[1,0]=1e-14
                ef,ea,ej=routing_metric(m,short,jacobian=True);eq=query_parameter(m)
                error=0.
                for j in range(12):
                    direction=torch.zeros_like(eq);direction[:,j]=1
                    _,derivative=torch.autograd.functional.jvp(lambda q:alpha_from_query(m,short,q),eq,direction,strict=True)
                    error=max(error,float(((derivative-ej[...,j]).abs()/(1+derivative.abs())).max()))
                assert error<1e-7 and torch.isfinite(ef).all() and torch.count_nonzero(ef[0])==0
                checks['real12_zero_and_sub_epsilon']=dict(relative_jvp_error=error,zero_fisher=True,finite=True)
            del m
        model.train();optimizer=make_optimizer(model,optcfg)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if mode!='quantum_adam':
            reference=copy.deepcopy(model)
            for (_,p),(_,rp) in zip(model.named_parameters(),reference.named_parameters()):rp.grad=p.grad.clone()
            plain=torch.optim.AdamW(reference.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01);plain.step()
            qname=next(n for n,p in model.named_parameters() if p is query_parameter(model))
            retained=before[qname]*(1-.001*.01)
            delta=(retained-query_parameter(reference).detach()).reshape(40,12).double().cpu().numpy()
            matrices=model.metric.cpu().numpy();directions=[]
            for matrix,d in zip(matrices,delta):
                values,vectors=np.linalg.eigh(matrix)
                damping=.01*np.trace(matrix)/12+1e-8
                shaped=vectors@((np.maximum(values,0)+damping)**(-.5)*(vectors.T@d))
                length=np.linalg.norm(d)
                directions.append(shaped*(length/max(np.linalg.norm(shaped),np.finfo(np.float64).tiny)))
            expected=retained.double().cpu().numpy()-np.array(directions).reshape(tuple(retained.shape))
        optimizer.step()
        if mode!='quantum_adam':
            error=float(np.abs(query_parameter(model).detach().cpu().numpy()-expected).max())
            assert error<5e-7
            assert all(torch.equal(p,rp) for (n,p),(_,rp) in zip(model.named_parameters(),reference.named_parameters()) if n!=qname)
            checks[mode+'_numpy_step']=dict(query_max_error=error,unchanged_body_step_exact=True)
            del reference
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        assert model.metric is None
        checks[mode+'_update']=dict(loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters()),
            finite=True,state_changed=True,query_step_check=getattr(optimizer,'last_check',None))
    return dict(status='PASS',checks=checks,real_first8_updates=len(models),
        scope='New query metric/derivatives/optimizer one batch only; no lifecycle/resume/COS matrix.')

def summarize(rows,logits,labels,cfg):
    for mode,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
    q=rows['quantum_fisher'];gates={}
    for mode,row in rows.items():
        if mode=='quantum_fisher':continue
        gates['macro_beats_'+mode]=q['all40_macro_f1']>row['all40_macro_f1']
        gates['ce_beats_'+mode]=q['ce']<row['ce']
    return dict(results=rows,gates=gates,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_FISHER_QUERY_SCREEN',
        interpretation='Exposed TRAIN4192/4000, fixed512 updates; Fisher-shaped norm-matched AdamW, not pure natural gradient or new theory. Short miss not family impossibility.',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
