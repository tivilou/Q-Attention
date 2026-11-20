"""Five prospective CUDA forward/autograd qualifications; no task training."""
import hashlib
import math
import torch
from torch.nn import functional as F
from experiments.rmea_context_geometry.legacy.rmea import EPS,RMEAHead
from experiments.rmea_utility_measurement.token_transport import transport_attention
from experiments.rmea_utility_measurement.token_transport_trainable import KINDS,route,TrainableTokenTransportHead

CHECKS=('TG-1-frozen-forward-parity','TG-2-routing-directional-derivatives',
        'TG-3-full-head-parameter-derivatives','TG-4-padding-permutation-and-input-domain',
        'TG-5-two-token-tie-and-single-token-boundary')


def state_hash(head):
    h=hashlib.sha256()
    for n,t in sorted(head.state_dict().items()):
        h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.manual_seed(20261120);torch.cuda.manual_seed_all(20261120)
    torch.use_deterministic_algorithms(True)
    dev=torch.device('cuda:0');residuals=[];gradient_errors=[];records=[]
    def rand(*shape):return torch.randn(shape,dtype=torch.float64,device=dev)
    def states(raw):
        z=torch.complex(raw[...,0],raw[...,1]);return z/z.norm(dim=-1,keepdim=True)
    def close(a,b):
        e=float((a-b).abs().max().detach());assert math.isfinite(e) and e<1e-9,e
        residuals.append(e);return e
    def derivative(ad,fd):
        error=abs(ad-fd);allowed=2e-9+2e-5*max(abs(ad),abs(fd),1e-6)
        assert math.isfinite(ad) and math.isfinite(fd) and error<=allowed,(ad,fd,error,allowed)
        gradient_errors.append((error,error/allowed))
        return dict(autograd=ad,finite_difference=fd,absolute_error=error,scaled_error=error/allowed)
    kr=rand(2,4,8,2).requires_grad_();qr=rand(40,8,2).requires_grad_()
    mask=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool,device=dev)
    mapping={'native':'native','cubic_coherent':'coherent','cubic_initial_dephased':'token_dephased'}
    parity=[]
    with torch.no_grad():
        psi=states(kr);phi=states(qr)
        for k,old in mapping.items():
            parity.append(dict(kind=k,residual=close(route(psi,phi,mask,kind=k)['alpha'],transport_attention(psi,phi,mask,kind=old)['alpha'])))
        linear=route(psi,phi,mask,kind='linear_coherent')['alpha']
        for b in range(2):
            ix=torch.where(mask[b])[0];s=psi[b,ix];g=s.conj()@s.T;k=g-torch.diag_embed(g.diagonal())
            u=torch.matrix_exp(-.25j*k/torch.linalg.eigvalsh(k).abs().max())
            amp=phi@s.conj().T;den=amp.abs().square().sum(-1)+len(ix)*EPS
            close(linear[b,:,ix],((amp@u.T).abs().square()+EPS)/den[:,None])
        original=RMEAHead('quantum').double().to(dev);initial={n:t.detach().clone() for n,t in original.state_dict().items()}
        assert sum(t.numel() for t in original.parameters())==81028
        features=rand(2,4,1024)*.1;roles=rand(2,4,4)*.1
        old=original(features,roles,mask,details=True)
        for k,oldkind in mapping.items():
            head=TrainableTokenTransportHead(k).double().to(dev);head.load_state_dict(initial)
            d=head(features,roles,mask,details=True);ref=transport_attention(old['psi'],old['phi'],mask,kind=oldkind)['alpha']
            close(d['alpha'],ref);close(d['logits'],(ref*old['utilities']).sum(-1)+original.bias)
            close(d['utilities'],old['utilities']);assert state_hash(head)==state_hash(original)
    records.append(dict(name=CHECKS[0],status='PASS',conditions=parity,all40logits_checked=True,original_parameters=81028))

    weights=rand(2,40,4);routing=[]
    for kind in KINDS:
        spectral_gap=None
        if kind!='native':
            spectrum=route(states(kr),states(qr),mask,kind=kind)['eigenvalues'].abs().sort(-1).values
            spectral_gap=float((spectrum[:,-1]-spectrum[:,-2]).min().detach())
            assert spectral_gap>1e-10,'Generic derivative fixture must have a unique spectral extremum'
        def scalar(x,y):return (route(states(x),states(y),mask,kind=kind)['alpha']*weights).sum()
        loss=scalar(kr,qr);grads=torch.autograd.grad(loss,(kr,qr))
        assert all(torch.isfinite(g).all() and float(g.norm())>1e-8 for g in grads)
        directional=[]
        for j in range(2):
            dx=rand(*kr.shape);dy=rand(*qr.shape);norm=(dx.square().sum()+dy.square().sum()).sqrt();dx=dx/norm;dy=dy/norm
            ad=float(((grads[0]*dx).sum()+(grads[1]*dy).sum()).detach())
            for step in (1e-5,5e-6):
                with torch.no_grad():fd=float((scalar(kr+step*dx,qr+step*dy)-scalar(kr-step*dx,qr-step*dy))/(2*step))
                directional.append(dict(direction=j,step=step,**derivative(ad,fd)))
        graph_path=None
        if kind!='native':
            d=route(states(kr),states(qr),mask,kind=kind);u=d['unitary'].detach()
            psi=states(kr).masked_fill(~mask[...,None],0);amp=torch.einsum('rd,btd->brt',states(qr),psi.conj())
            den=amp.abs().square().sum(-1)+mask.sum(-1)[:,None].to(amp.real.dtype)*EPS
            nat=(amp.abs().square()+EPS)*mask[:,None,:]/den[...,None]
            if kind=='cubic_initial_dephased':partial=nat@u.abs().square().transpose(-2,-1)
            else:partial=((amp@u.transpose(-2,-1)).abs().square()+EPS)/den[...,None]
            partial=(partial*mask[:,None,:]*weights).sum()
            fixed_grad=torch.autograd.grad(partial,kr)[0]
            graph_path=float((grads[0]-fixed_grad).norm().detach());assert graph_path>1e-8
        routing.append(dict(kind=kind,derivatives=directional,key_gradient_norm=float(grads[0].norm()),query_gradient_norm=float(grads[1].norm()),nonzero_gradient_through_graph=graph_path,minimum_absolute_spectral_gap=spectral_gap))
    records.append(dict(name=CHECKS[1],status='PASS',conditions=routing,unique_spectral_extremum_scope=True))

    labels=torch.tensor([2,17],device=dev);head_checks=[]
    groups={'prepare':('prepare.weight','prepare.bias'),'effect':('effect',),'value':('value.weight','value.bias'),'decoder':('decoder',),'bias':('bias',)}
    directions={n:rand(*t.shape) for n,t in original.named_parameters()}
    for names in groups.values():
        norm=sum(directions[n].square().sum() for n in names).sqrt()
        for n in names:directions[n]=directions[n]/norm
    for kind in KINDS:
        head=TrainableTokenTransportHead(kind).double().to(dev);head.load_state_dict(initial);before=state_hash(head)
        params=dict(head.named_parameters());logits,alpha=head(features,roles,mask);loss=F.cross_entropy(logits,labels)
        grads=torch.autograd.grad(loss,tuple(params.values()));gd=dict(zip(params,grads))
        assert all(torch.isfinite(g).all() for g in grads)
        checked=[]
        for group,names in groups.items():
            norm=float(sum(gd[n].square().sum() for n in names).sqrt());assert norm>1e-8,(kind,group,norm)
            ad=float(sum((gd[n]*directions[n]).sum() for n in names).detach());step=1e-4
            with torch.no_grad():
                plus={n:p+(step*directions[n] if n in names else 0) for n,p in params.items()}
                minus={n:p-(step*directions[n] if n in names else 0) for n,p in params.items()}
                lp=F.cross_entropy(torch.func.functional_call(head,plus,(features,roles,mask),strict=True)[0],labels)
                lm=F.cross_entropy(torch.func.functional_call(head,minus,(features,roles,mask),strict=True)[0],labels)
                fd=float((lp-lm)/(2*step))
            checked.append(dict(group=group,gradient_norm=norm,**derivative(ad,fd)))
        assert state_hash(head)==before==state_hash(original)
        head_checks.append(dict(kind=kind,cross_entropy=float(loss.detach()),groups=checked,state_unchanged=True))
    records.append(dict(name=CHECKS[2],status='PASS',conditions=head_checks,optimizer_steps=0))

    with torch.no_grad():
        psi=states(kr);phi=states(qr);permutation=torch.tensor([2,0,3,1],device=dev)
        for kind in KINDS:
            d=route(psi,phi,mask,kind=kind)
            close(route(psi[:,permutation],phi,mask[:,permutation],kind=kind)['alpha'],d['alpha'][:,:,permutation])
            padded=torch.cat((psi,torch.full((2,2,8),complex(float('nan'),0),dtype=psi.dtype,device=dev)),1)
            pm=torch.cat((mask,torch.zeros((2,2),dtype=torch.bool,device=dev)),1)
            pad=route(padded,phi,pm,kind=kind)['alpha'];close(pad[:,:,:4],d['alpha']);assert (pad[:,:,4:]==0).all()
            close(d['alpha'].sum(-1),torch.ones((2,40),dtype=torch.float64,device=dev))
        cases=[(psi,phi,torch.zeros_like(mask),'native'),(psi.to(torch.complex64),phi,mask,'native')]
        orth=torch.eye(8,dtype=torch.complex128,device=dev)[:2][None];m2=torch.ones((1,2),dtype=torch.bool,device=dev)
        cases.extend((orth,phi,m2,k) for k in KINDS if k!='native')
        rejected=0
        for s,p,m,k in cases:
            try:route(s,p,m,kind=k)
            except ValueError:rejected+=1
            else:raise AssertionError('Invalid input accepted')
        assert rejected==5
        bad_features=features.clone();bad_features[1,3,0]=float('nan')
        try:head(bad_features,roles,mask)
        except ValueError:pass
        else:raise AssertionError('Full head must reject nonfinite feature padding')
    records.append(dict(name=CHECKS[3],status='PASS',masked_nonfinite_padding_ignored=True,permutation_equivariance=True,invalid_inputs_rejected=rejected,zero_multitoken_graph_rejected=True,full_head_nonfinite_feature_padding_rejected=True))

    x=rand(1,2,8,2).requires_grad_();p=states(qr.detach());m2=torch.ones((1,2),dtype=torch.bool,device=dev);w=rand(1,40,2)
    c=route(states(x),p,m2,kind='cubic_coherent');l=route(states(x),p,m2,kind='linear_coherent');close(c['alpha'],l['alpha'])
    tied=[]
    for kind in ('cubic_coherent','cubic_initial_dephased','linear_coherent'):
        def f(raw):return (route(states(raw),p,m2,kind=kind)['alpha']*w).sum()
        grad=torch.autograd.grad(f(x),x)[0];dx=rand(*x.shape);dx=dx/dx.norm();ad=float((grad*dx).sum().detach());step=1e-5
        with torch.no_grad():fd=float((f(x+step*dx)-f(x-step*dx))/(2*step))
        tied.append(dict(kind=kind,**derivative(ad,fd)))
        one=route(states(x)[:,:1],p,m2[:,:1],kind=kind)['alpha'];close(one,torch.ones_like(one))
    with torch.no_grad():
        basis=torch.eye(8,dtype=torch.complex128,device=dev)
        singleton=basis[0][None,None];orthogonal_queries=basis[1].expand(40,-1).clone()
        singleton_mask=torch.ones((1,1),dtype=torch.bool,device=dev);orthogonal_errors=[]
        for kind in KINDS:
            result=route(singleton,orthogonal_queries,singleton_mask,kind=kind)['alpha']
            oldkind=mapping.get(kind,'native')
            reference=transport_attention(singleton,orthogonal_queries,singleton_mask,kind=oldkind)['alpha']
            orthogonal_errors.append(dict(kind=kind,residual=close(result,reference)))
            assert torch.equal(result,torch.ones_like(result)),'Exact orthogonal singleton normalization failed'
    records.append(dict(name=CHECKS[4],status='PASS',two_token_cubic_linear_parity=True,restricted_two_token_derivatives=tied,single_token_identity=True,orthogonal_singleton_forward_oracle=orthogonal_errors,arbitrary_nondifferentiable_ties_not_certified=True))
    def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    fixture=dict(seed=20261120,initial_head_sha256=state_hash(original),source_deterministic_algorithms=True,
                 tensors={n:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=tensor_hash(t)) for n,t in dict(raw_keys=kr,raw_queries=qr,features=features,roles=roles,mask=mask,routing_weights=weights).items()},sample_trace_v2_certified=False)
    return dict(status='TOKEN_TRANSPORT_GRADIENT_CHECK_COMPLETE',check_count=5,checks=records,fixture=fixture,max_absolute_residual=max(residuals),max_gradient_absolute_error=max(v[0] for v in gradient_errors),max_gradient_scaled_error=max(v[1] for v in gradient_errors),device=dict(type='cuda',name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda),optimizer_steps=0,real_dataset_reads=0,new_checkpoints=0,COS_requests=0,cloud_writes=0,task_training=False,independent_task_confirmation=False,formal_handoff=False,L5='NOT_ACCEPTED',limitations=['Directional derivatives qualify frozen generic fixtures, not every input or global differentiability.','Nonzero spectral ties use an autograd subgradient; zero multi-token graph fails closed.','No task training or label-benefit evidence.'])
