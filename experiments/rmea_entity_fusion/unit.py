"""Independent physical-channel and boundary checks before a bounded screen."""
import torch
from fusion import FusionHead,fuse,VARIANTS
from experiments.rmea_allocation.legacy.rmea import RMEAHead

def density(d):
    x=torch.randn(d,d,dtype=torch.complex128);x=x@x.mH;return x/x.trace()

def partial_swap(a,b,kappa,lam):
    """Independent d^2 unitary construction plus explicit index partial trace."""
    d=len(a);swap=torch.zeros(d*d,d*d,dtype=a.dtype)
    for i in range(d):
        for j in range(d):swap[j*d+i,i*d+j]=1
    weight=(1-kappa)/2;radius=(1-kappa*kappa)**.5
    tensor=torch.kron(a.contiguous(),b.contiguous());eye=torch.eye(d*d,dtype=a.dtype)
    out=[]
    for sign in [1,-1]:
        u=weight**.5*eye+sign*1j*(1-weight)**.5*swap
        full=(u@tensor@u.mH).reshape(d,d,d,d)
        traced=torch.zeros_like(a)
        for i in range(d):
            for j in range(d):
                for q in range(d):traced[i,j]+=full[i,q,j,q]
        out.append(traced)
    p=(1-lam/radius)/2
    return p*out[0]+(1-p)*out[1]

def run():
    torch.manual_seed(916);maximum=0.;cases=0
    for d in [2,8]:
        for _ in range(12):
            a,b=density(d),density(d);g=torch.randn(2,dtype=torch.float64);g=.9*g/torch.sqrt(1+g.square().sum());k,l=g
            e=fuse(a,b,k,l);ref=partial_swap(a,b,float(k),float(l))
            err=float((e-ref).abs().max());maximum=max(maximum,err);assert err<2e-12
            assert (e-e.mH).abs().max()<2e-12 and abs(e.trace()-1)<2e-12
            assert torch.linalg.eigvalsh(e).min()>-2e-12 and torch.linalg.eigvalsh(e).max()<1+2e-12
            cases+=1
    a=torch.diag(torch.tensor([1.,0.],dtype=torch.complex128));v=torch.tensor([.99**.5,.01**.5],dtype=a.dtype);b=v[:,None]*v.conj()[None]
    assert torch.linalg.eigvalsh(fuse(a,b,torch.tensor(.8),torch.tensor(.8))).min()<0
    # Nonzero commutator, identical valid-token states: normalization erases it.
    plus=torch.tensor([1.,1.],dtype=a.dtype)/2**.5;b=plus[:,None]*plus.conj()[None]
    imag=torch.tensor([1.,1j],dtype=a.dtype)/2**.5;rho=imag[:,None]*imag.conj()[None]
    assert abs(torch.trace(.5j*(a@b-b@a)@rho))>.2
    aa=[]
    for lam in [-1.,0.,1.]:
        e=fuse(a,b,torch.tensor(0.),torch.tensor(lam));p=torch.trace(e@rho).real.repeat(3)+1e-6;aa.append(p/p.sum())
    assert max(float((x-aa[0]).abs().max()) for x in aa)<1e-12
    # Check a physical score derivative against finite differences.
    coeff=torch.tensor([.2,.3],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda g:torch.trace(fuse(a,b,g[0],g[1])@rho).real,(coeff,),eps=1e-6,atol=1e-6,rtol=1e-4)
    h=torch.randn(2,5,1024,dtype=torch.float64);roles=torch.zeros(2,5,4,dtype=torch.float64);mask=torch.tensor([[1]*5,[1,1,1,0,0]],dtype=torch.bool)
    for row,n in enumerate([5,3]):
        t=torch.arange(5,dtype=torch.float64);roles[row,:,0]=t/32;roles[row,:,1]=(t-(n-1))/32;roles[row,0,2]=1;roles[row,n-1,3]=1
    shared=None;parameter_counts={};gradients={}
    for name in VARIANTS:
        torch.manual_seed(307);m=FusionHead(name).double();state={n:p.detach().clone() for n,p in m.named_parameters() if not n.startswith(('gates','effect'))}
        if shared is None:shared=state
        else:assert state.keys()==shared.keys() and all(torch.equal(state[n],shared[n]) for n in shared)
        parameter_counts[name]=sum(p.numel() for p in m.parameters())
        d=m(h,roles,mask,details=True);assert torch.isfinite(d['logits']).all() and (d['alpha'].sum(-1)-1).abs().max()<1e-12
        assert d['alpha'].masked_select((~mask)[:,None].expand_as(d['alpha'])).abs().max()==0
        padded=m(torch.cat((h,torch.ones(2,2,1024,dtype=h.dtype)*7),1),torch.cat((roles,torch.zeros(2,2,4,dtype=h.dtype)),1),torch.cat((mask,torch.zeros(2,2,dtype=torch.bool)),1))[0]
        assert (d['logits']-padded).abs().max()<1e-10
        torch.nn.functional.cross_entropy(d['logits'],torch.tensor([1,2])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in m.parameters())
        gradients[name]={n:float(p.grad.abs().max()) for n,p in m.named_parameters()}
        if name=='global':
            torch.manual_seed(307);legacy=RMEAHead('quantum').double()
            assert all(torch.equal(v,legacy.state_dict()[k]) for k,v in m.state_dict().items())
            zz,al=legacy(h,roles,mask);torch.testing.assert_close(zz,d['logits'],atol=1e-12,rtol=1e-12);torch.testing.assert_close(al,d['alpha'],atol=1e-12,rtol=1e-12)
            torch.nn.functional.cross_entropy(zz,torch.tensor([1,2])).backward()
            for n,p in m.named_parameters():torch.testing.assert_close(p.grad,dict(legacy.named_parameters())[n].grad,atol=1e-12,rtol=1e-12)
        if name=='fusion':
            assert torch.equal(d['logits'],m(h,roles,mask,phase_off=True)[0])
        bad=roles.clone();bad[:,:,2]=0
        try:m(h,bad,mask)
        except ValueError:pass
        else:raise AssertionError('Empty entity accepted')
    assert parameter_counts=={'global':81028,'even':80548,'mixture':80588,'fusion':80628}
    return dict(status='PASS',physical_channel_cases=cases,maximum_channel_error=maximum,parameter_counts=parameter_counts,shared_initialization_equal=True,normalization_cancellation=True,padding_invariance=True,finite_gradients=gradients,scope='Physical channel correspondence and numerical boundaries, not task utility or L5')
