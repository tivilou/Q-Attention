"""Changed multi-route checks, fixed evaluation and one tying intervention."""
from pathlib import Path
import copy,hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES,initialize,make_head,marginals,routing,readout
from math_check import direct_probabilities

def direct_scores(states,queries,dimensions,population):
    a,b,c=dimensions
    ss=(states.reshape(states.shape[:-1]+(a*b,c)),states.reshape(states.shape[:-1]+(a,b*c)))
    qq=(queries.reshape(queries.shape[:-1]+(a*b,c)),queries.reshape(queries.shape[:-1]+(a,b*c)))
    values=[]
    for k,(s,q) in enumerate(zip(ss,qq)):
        if population:
            sa=s.abs().square().sum(-1 if k==0 else -2)
            qa=q.abs().square().sum(-1 if k==0 else -2)
            values.append(torch.einsum('ri,bti->brt',qa,sa))
        else:
            inner=torch.einsum('ric,btid->brtcd',q.conj(),s) if k==0 else torch.einsum('rai,btci->brtac',q.conj(),s)
            values.append(inner.abs().square().sum((-1,-2)))
    return torch.stack(values,-1)

def fixtures(models,inputs,labels):
    result={}
    def check(value,tol=1e-9):
        value=float(value);assert np.isfinite(value) and value<tol,(value,tol);return value
    for mode,original in models.items():
        model=copy.deepcopy(original).to(inputs[0].device);assert sum(p.numel() for p in model.parameters())==81028
        double=copy.deepcopy(model).double();n=min(7,inputs[0].shape[1]);assert n>=2
        x=(inputs[0][:2,:n].double(),inputs[1][:2,:n].double(),torch.ones((2,n),dtype=torch.bool,device=inputs[0].device))
        states,queries,dims,alpha,utility=double.components(*x);population=mode=='population_marginal_split'
        p=direct_scores(states,queries,dims,population).clamp_min(0)
        raw=(p+1e-6)*x[2][:,None,:,None];explicit_alpha=raw/raw.sum(-2,keepdim=True)
        formula=check((alpha-explicit_alpha).abs().max())
        reference=readout(explicit_alpha,utility,double.head.bias,mode=='quantum_marginal_tied');got=double(*x)
        check((reference-got).abs().max())
        params=tuple(double.parameters());ga=torch.autograd.grad(got.square().mean(),params,retain_graph=True)
        gb=torch.autograd.grad(reference.square().mean(),params)
        gradient=check(max((a-b).abs().max() for a,b in zip(ga,gb)),1e-7)
        with torch.no_grad():
            expected=np.stack([direct_probabilities(states[b,:2].detach().cpu().numpy(),queries[:2].detach().cpu().numpy(),dims,population) for b in range(2)])
            numpy_error=check(np.max(abs(expected-p[:,:2,:2].cpu().numpy())))
            restored=make_head(mode).to(device=inputs[0].device,dtype=torch.float64);restored.load_state_dict(double.state_dict(),strict=True)
            factory=check((restored(*x)-got).abs().max())
            padded=(F.pad(x[0],(0,0,0,3),value=.2),F.pad(x[1],(0,0,0,3),value=1),F.pad(x[2],(0,3),value=False))
            padding=check((double(*padded)-got).abs().max())
            one=(x[0][:1,:1],x[1][:1,:1],x[2][:1,:1]);native=(double.head.value(one[0])[:,0]*double.head.decoder).sum(-1)+double.head.bias
            check((double(*one)-native).abs().max())
            phase=torch.exp(1j*torch.linspace(-.8,.9,n,dtype=torch.float64,device=states.device)) if states.is_complex() else torch.where(torch.arange(n,device=states.device)%2==0,1.,-1.).double()
            qphase=torch.exp(1j*torch.linspace(-.3,.7,40,dtype=torch.float64,device=states.device)) if states.is_complex() else torch.where(torch.arange(40,device=states.device)%2==0,1.,-1.).double()
            gauge=check((routing(states*phase[None,:,None],queries*qphase[:,None],dims,x[2],population)-alpha).abs().max())
            separate=readout(alpha,utility,double.head.bias);tied=readout(alpha,utility,double.head.bias,True)
            delta=.5*((alpha[...,0]-alpha[...,1])*(utility[...,0]-utility[...,1])).sum(-1)
            identity=check((separate-tied-delta).abs().max())
            for density in (*marginals(states,dims),*marginals(queries,dims)):
                check((density.diagonal(dim1=-2,dim2=-1).sum(-1)-1).abs().max())
                assert torch.linalg.eigvalsh(density).min()>-1e-9
            assert p.min()>=0 and p.max()<=1+1e-9
        for bad in ((x[0],x[1],torch.zeros_like(x[2])),tuple(t[:0] for t in x)):
            try:double(*bad)
            except ValueError:pass
            else:raise AssertionError('Invalid input accepted')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        before={n:v.detach().clone() for n,v in model.named_parameters()}
        loss=F.cross_entropy(model(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(v).all() for v in model.parameters()) and any(not torch.equal(before[n],v) for n,v in model.named_parameters())
        result[mode]=dict(parameters=81028,formula_error=formula,gradient_error=gradient,numpy_error=numpy_error,padding_error=padding,gauge_error=gauge,readout_identity_error=identity,factory_roundtrip_error=factory,real8_ce=float(loss.detach()),update=True)
    for mode in ('population_marginal_split','quantum_marginal_tied'):
        assert all(torch.equal(v,models[mode].state_dict()[n]) for n,v in models['quantum_marginal_split'].state_dict().items())
    return dict(status='PASS',checks=result,scope='New local-state routing/value grouping; cloned8 real updates, stable core/resume/COS unchanged')

def margin(z,y):
    out=z.astype(np.float64).copy();gold=out[np.arange(len(y)),y].copy();out[np.arange(len(y)),y]=-np.inf
    return gold-out.max(-1)

def intervention(rows,logits,labels,cfg):
    start=time.monotonic();spec=cfg['intervention'];folder=Path(spec['output']);timeout=float(spec['timeout_seconds'])
    private=folder/'diagnostic.private.npz'
    with np.load(private,allow_pickle=False) as values:
        assess=values['assessment_indices'].copy();all_labels=values['labels'].copy()
        assert np.array_equal(labels,all_labels[assess]) and np.array_equal(logits['quantum_marginal_split'],values['quantum_marginal_split_logits'])
    final=folder/'quantum_marginal_split/final.pt'
    assert hashlib.sha256(final.read_bytes()).hexdigest()==rows['quantum_marginal_split']['final_sha256']
    cache=Path(spec['cache']);data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    assert np.array_equal(labels,data['targets'][assess])
    device=torch.device('cuda',0);model=make_head('quantum_marginal_split').to(device=device,dtype=torch.float32)
    model.load_state_dict(torch.load(final,map_location=device,weights_only=True),strict=True);model.eval()
    intact=[];tied=[]
    with torch.no_grad():
        for lo in range(0,len(assess),128):
            assert time.monotonic()-start<timeout
            ids=assess[lo:lo+128];lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
            h=np.zeros((len(ids),max(lengths),1024),np.float32);roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
            for j,i in enumerate(ids):
                a,b=map(int,data['offsets'][i:i+2]);h[j,:b-a]=data['h'][a:b];roles[j,:b-a]=data['roles'][a:b];mask[j,:b-a]=True
            inputs=tuple(torch.from_numpy(v).to(device) for v in (h,roles,mask))
            _,_,_,alpha,utility=model.components(*inputs)
            intact.append(readout(alpha,utility,model.head.bias).cpu().numpy())
            tied.append(readout(alpha,utility,model.head.bias,True).cpu().numpy())
    intact=np.concatenate(intact);tied=np.concatenate(tied)
    assert time.monotonic()-start<timeout,'Replay deadline exceeded after final synchronized CPU copy'
    error=float(np.max(abs(intact-logits['quantum_marginal_split'])));assert error<=3e-5
    positive=labels>0;assert positive.any();difference=margin(intact,labels)-margin(tied,labels)
    assert np.isfinite(difference).all()
    path=folder/'intervention.private.npz';np.savez(path,indices=assess,labels=labels,intact=intact,tied=tied)
    elapsed=time.monotonic()-start;assert elapsed<timeout,'Replay deadline exceeded before returning gate evidence'
    return dict(positive_rows=int(positive.sum()),mean_positive_margin_difference=float(difference[positive].mean()),mean_all_margin_difference=float(difference.mean()),original_replay_max_error=error,seconds=elapsed,private_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),largest_rival_recomputed=True,all_fits_before_replay=True)

def summarize(rows,logits,labels,cfg):
    predictions={}
    for mode,z in logits.items():
        pred=z.argmax(-1);predictions[mode]=pred;cm=np.bincount(40*labels+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[mode].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),class_support=cm.sum(1).tolist(),per_class_f1=f.tolist())
    refs=cfg['reused_native_rows'];assert set(refs)=={'quantum_native','real12','softmax12','rbf12'}
    q=rows['quantum_marginal_split'];assert {v['order_sha256'] for v in rows.values()}=={v['order_sha256'] for v in refs.values()}
    assert all(v['class_support']==q['class_support'] and v['updates']==512 for v in refs.values())
    others={k:v for k,v in rows.items() if k!='quantum_marginal_split'};others.update({'reused/'+k:v for k,v in refs.items()})
    gates={'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in others.items()};assert len(gates)==7
    check=intervention(rows,logits,labels,cfg)
    gates['positive_margin_beats_frozen_tying']=check['mean_positive_margin_difference']>0
    correct=predictions['quantum_marginal_split']==labels;paired={}
    for mode,pred in predictions.items():
        if mode=='quantum_marginal_split':continue
        peer=pred==labels;paired[mode]=dict(repair=int((correct&~peer).sum()),damage=int((~correct&peer).sum()),both_correct=int((correct&peer).sum()),both_wrong=int((~correct&~peer).sum()))
    return dict(results=rows,reused_results=refs,gates=gates,intervention=check,paired_outcomes=paired,gate='PASS_FOR_FULL_PROTOCOL_ONLY' if all(gates.values()) else 'CLOSE_EXACT_MARGINAL_ROUTING_SCREEN',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',interpretation='Known local-state/multi-head tools; exposed pilot, no quantum-specific cause or newT accepted')
