"""Affected product probabilities/gradients and the six primary Macro gates."""
import copy,math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit
from baselines.retacred.rmea_matched_routers import parameter_schema
from plugin import MODES,BASE,RegisterHead,initialize,qubit_state,product_probability

def make_head(mode):return RegisterHead(mode)

def checked_error(value,tolerance):
    error=float(value);assert np.isfinite(error) and error<tolerance,error
    return error

def tensor_state(angles):
    one=qubit_state(angles);v=one[...,0,:]
    for k in range(1,6):v=(v[...,None]*one[...,k,None,:]).flatten(-2)
    return v

def numpy_state(angles):
    v=np.ones(1,dtype=np.complex128)
    for y,z in angles:v=np.kron(v,[np.cos(y/2)*np.exp(-.5j*z),np.sin(y/2)*np.exp(.5j*z)])
    return v

def ring_permutation():
    p=list(range(64))
    for c,t in ((0,1),(1,2),(2,3),(3,4),(4,5),(5,0)):
        step=[i^((1<<(5-t)) if (i>>(5-c))&1 else 0) for i in range(64)]
        p=[p[i] for i in step]
    return p

def fixtures(models,inputs,labels):
    device=inputs[0].device;checks={}
    key=torch.linspace(-1.4,1.2,36,dtype=torch.float64,device=device).reshape(1,3,6,2).requires_grad_(True)
    query=torch.linspace(.9,-1.1,48,dtype=torch.float64,device=device).reshape(4,6,2).requires_grad_(True)
    ks=tensor_state(key);qs=tensor_state(query)
    numk=np.stack([numpy_state(v) for v in key.detach().cpu().numpy().reshape(-1,6,2)])[None]
    numq=np.stack([numpy_state(v) for v in query.detach().cpu().numpy()])
    checks['numpy_state_error']=checked_error((ks-torch.tensor(numk,device=device)).abs().max(),1e-12)
    perm=ring_permutation();w=torch.linspace(-.8,1.3,12,dtype=torch.float64,device=device).reshape(1,4,3)
    for dephased in (False,True):
        p=product_probability(key,query,dephased)
        nk=numk[...,perm];nq=numq[...,perm]
        reference=(np.einsum('rd,btd->brt',np.abs(nq)**2,np.abs(nk)**2) if dephased else
                   np.abs(np.einsum('rd,btd->brt',nq.conj(),nk))**2)
        error=checked_error((p-torch.tensor(reference,device=device)).abs().max(),1e-12)
        tf=(torch.einsum('rd,btd->brt',qs[...,perm].abs().square(),ks[...,perm].abs().square()) if dephased else
            torch.einsum('rd,btd->brt',qs[...,perm].conj(),ks[...,perm]).abs().square())
        gp=torch.autograd.grad((p*w).sum(),(key,query),retain_graph=True)
        gt=torch.autograd.grad((tf*w).sum(),(key,query),retain_graph=True)
        ge=max(checked_error((a-b).abs().max(),1e-11) for a,b in zip(gp,gt))
        checks[('dephased' if dephased else 'quantum')+'_product']=dict(numpy_probability_error=error,all_key_query_weighted_gradient_error=ge)
    # A finite-parameter 40-code construction, not a claim that text learns it.
    codes=torch.zeros((40,6,2),dtype=torch.float64,device=device)
    for i in range(40):
        for k in range(6):codes[i,k,0]=math.pi*(.5 if (i>>k)&1 else -.5)
    assert torch.isfinite(torch.atanh(codes/math.pi)).all()
    state=tensor_state(codes);gram=(state@state.conj().T).abs().square()
    checks['finite40_orthogonal_error']=checked_error((gram-torch.eye(40,device=device)).abs().max(),1e-12)
    native=models['quantum_native'].head
    nq=circuit(math.pi*torch.tanh(native.effect).double())
    ng=(nq@nq.conj().T).abs().square();off=(ng.sum()-ng.diag().sum())/(40*39)
    assert torch.isfinite(ng).all() and off>=(40-8)/(8*39)-1e-10
    checks['native_initial_mean_offdiagonal']=float(off);checks['known_native_welch_lower_bound']=(40-8)/(8*39)
    for mode,original in models.items():
        schema=parameter_schema(BASE[mode]);state=original.native.state_dict()
        assert set(state)==set(schema)
        assert all(list(v.shape)==schema[n] and v.dtype==torch.float32 for n,v in state.items())
        model=copy.deepcopy(original).to(device);assert sum(p.numel() for p in model.parameters())==81028
        short=tuple(v.double() if v.dtype!=torch.bool else v for v in inputs)
        dm=copy.deepcopy(model).double()
        with torch.no_grad():
            z=dm(*short);assert torch.isfinite(z).all()
            padded=(F.pad(short[0],(0,0,0,2),value=.123),F.pad(short[1],(0,0,0,2),value=1),F.pad(short[2],(0,2),value=False))
            padding=checked_error((z-dm(*padded)).abs().max(),1e-10)
            if mode in ('quantum_native','real12','softmax12'):assert torch.equal(z,dm.native(*short))
            else:
                alpha=dm.route_alpha(*short)
                assert torch.isfinite(alpha).all() and alpha.min()>=0
                checked_error((alpha.sum(-1)-1).abs().max(),1e-12)
                assert torch.count_nonzero(alpha.masked_select(~short[2][:,None].expand_as(alpha)))==0
                for bad_mask in (torch.zeros_like(short[2]),short[2].double(),short[2][:1]):
                    try:dm.route_alpha(short[0],short[1],bad_mask)
                    except ValueError:pass
                    else:raise AssertionError('Invalid mask was accepted')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
        checks[mode+'_update']=dict(parameters=81028,loss=float(loss.detach()),padding_error=padding,finite=True,state_changed=True)
    base=models['quantum_native'].state_dict()
    assert all(torch.equal(v,models[m].state_dict()[n]) for m in ('quantum_register6','dephased_register6') for n,v in base.items())
    return dict(status='PASS',checks=checks,scope='Changed register/product kernel plus fixed math witness and one cloned real8 update/arm; no lifecycle matrix.')

def summarize(rows,logits,labels,cfg):
    for mode,z in logits.items():
        pred=z.argmax(-1);cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            positive_wrong_relation=int(((labels>0)&(pred>0)&(labels!=pred)).sum()))
    q=rows['quantum_register6'];gates={'macro_beats_'+m:q['all40_macro_f1']>v['all40_macro_f1'] for m,v in rows.items() if m!='quantum_register6'}
    new=logits['quantum_register6'].argmax(-1);paired={}
    for mode,z in logits.items():
        if mode=='quantum_register6':continue
        old=z.argmax(-1);repair=(new==labels)&(old!=labels);harm=(new!=labels)&(old==labels)
        paired[mode]=dict(repair=int(repair.sum()),harm=int(harm.sum()),net_correct=int(repair.sum()-harm.sum()),
            repair_by_gold=np.bincount(labels[repair],minlength=40).tolist(),harm_by_gold=np.bincount(labels[harm],minlength=40).tolist())
    return dict(results=rows,gates=gates,paired=paired,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_REGISTER_CAPACITY_SCREEN',
        interpretation='Known packing/product-kernel tools, not text-cause, hardware advantage, independentL3 or newT; CE secondary.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
