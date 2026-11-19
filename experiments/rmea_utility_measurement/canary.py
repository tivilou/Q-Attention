"""Finite, predeclared numerical canary. Run only in its shared CPU recipe."""
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead, circuit
from experiments.rmea_utility_measurement.head import UtilityMeasurementHead, measured_attention, utility_permutation


def reference(psi, phi, u, mask, eta=1.):
    """Independent NumPy spectral forward; not used for derivatives."""
    b, t, d = psi.shape
    result = np.zeros((b, len(phi), t))
    effects = np.zeros((b, len(phi), d, d), dtype=np.complex128)
    for i in range(b):
        rho = [np.outer(psi[i,j], psi[i,j].conj()) if mask[i,j] else np.zeros((d,d), complex) for j in range(t)]
        s = sum(rho)
        for r in range(len(phi)):
            p = np.outer(phi[r], phi[r].conj())
            e0 = (p + EPS*np.eye(d))/(1+d*EPS)
            # Unlike production's projector formula, obtain log(E0) spectrally.
            w, v = np.linalg.eigh(e0)
            log0 = (v*np.log(w))@v.conj().T
            op = sum(u[i,r,j]*rho[j] for j in range(t))
            f0 = np.trace(e0@op).real/np.trace(e0@s).real
            a = op-f0*s
            h = log0+eta*a/np.sqrt(1+np.linalg.norm(a,'fro')**2)
            w, v = np.linalg.eigh(h)
            e = (v*np.exp(w))@v.conj().T
            e /= np.trace(e).real
            effects[i,r] = e
            raw = np.asarray([np.trace(e@rt).real for rt in rho])
            result[i,r] = raw/raw.sum()
    return result, effects


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261119)
    checks = []
    def record(name, **values):
        checks.append(dict(name=name, status='PASS', **values))
    def close(a,b,tol=2e-10):
        assert torch.allclose(a,b,atol=tol,rtol=0),float((a-b).abs().max())
    def states(shape):
        z = torch.randn(shape,dtype=torch.float64)+1j*torch.randn(shape,dtype=torch.float64)
        return z/z.norm(dim=-1,keepdim=True)
    psi, phi = states((2,4,8)), states((3,8))
    u = torch.randn((2,3,4),dtype=torch.float64)
    mask = torch.tensor([[True,True,True,False],[True,True,False,False]])
    out = measured_attention(psi,phi,u,mask)
    ra,re = reference(psi.numpy(),phi.numpy(),u.numpy(),mask.numpy())
    close(out['alpha'],torch.from_numpy(ra));close(out['effect'],torch.from_numpy(re))
    record('independent_numpy_forward',max_alpha_error=float((out['alpha']-torch.from_numpy(ra)).abs().max()))
    p = torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
    raw = (p+EPS)*mask[:,None,:];a0=raw/raw.sum(-1,keepdim=True)
    zero = measured_attention(psi,phi,u,mask,eta=0)
    close(zero['alpha'],a0)
    record('eta_zero_original_born',max_alpha_error=float((zero['alpha']-a0).abs().max()))
    eig = torch.linalg.eigvalsh(out['effect'])
    assert float(eig.min())>0 and float(eig.max())<=1+2e-10
    close(out['alpha'].sum(-1),torch.ones((2,3),dtype=torch.float64))
    assert torch.equal(out['alpha'].masked_select(~mask[:,None,:]),torch.zeros(9,dtype=torch.float64))
    record('positive_normalized_masked',min_effect_eigenvalue=float(eig.min()))
    diff=out['aggregate']-zero['aggregate'];assert float(diff.min())>=-2e-10
    divergences=[]
    for i in range(2):
        for r in range(3):
            e=out['effect'][i,r].numpy();w,v=np.linalg.eigh(e);loge=(v*np.log(w))@v.conj().T
            drel=np.trace(e@(loge-out['initial_log'][r].numpy())).real
            denom=np.trace(e@sum(np.outer(psi[i,j].numpy(),psi[i,j].numpy().conj()) for j in range(4) if mask[i,j])).real
            assert float(diff[i,r])+2e-10>=float(out['scale'][i,r])*drel/denom
            divergences.append(float(drel))
    record('gibbs_bound',minimum_own_score_gain=float(diff.min()),minimum_relative_entropy=min(divergences))
    close(measured_attention(psi,phi,u+.73,mask)['alpha'],out['alpha'])
    q,_=torch.linalg.qr(states((8,8)))
    close(measured_attention(psi@q.T,phi@q.T,u,mask)['alpha'],out['alpha'])
    phase=torch.exp(1j*torch.tensor([[.2,.5,-.3,.1],[-.9,.4,.6,.2]],dtype=torch.float64))
    close(measured_attention(psi*phase[...,None],phi,u,mask)['alpha'],out['alpha'])
    record('utility_shift_unitary_and_token_phase_covariance')
    badpsi=psi.clone();badu=u.clone();badpsi[~mask]=complex(float('nan'),0);badu.masked_fill_(~mask[:,None,:],float('nan'))
    close(measured_attention(badpsi,phi,badu,mask)['alpha'],out['alpha'])
    record('padding_nan_excluded_not_clipped')
    # Positive rescaling is smooth, deliberately NOT invariant in v002.
    scale_reports=[]
    a=out['covariance'];a2=a.abs().square().sum((-1,-2))
    for factor in [.5, 1.-1e-4, 1., 1.+1e-4, 2.]:
        z=measured_attention(psi,phi,u*factor,mask)
        close(z['covariance'],factor*a)
        close(z['scale'],torch.sqrt(1+factor*factor*a2))
        close(z['covariance']/z['scale'][...,None,None],factor*a/torch.sqrt(1+factor*factor*a2)[...,None,None])
        rr,ee=reference(psi.numpy(),phi.numpy(),(u*factor).numpy(),mask.numpy())
        close(z['alpha'],torch.from_numpy(rr));close(z['effect'],torch.from_numpy(ee))
        scale_reports.append(dict(factor=factor,max_alpha_change=float((z['alpha']-out['alpha']).abs().max())))
    sv=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    aw=torch.arange(u.numel(),dtype=torch.float64).reshape_as(u)/u.numel()
    def scaling_objective(v):return (measured_attention(psi,phi,u*v,mask)['alpha']*aw).sum()
    sg=torch.autograd.grad(scaling_objective(sv),sv)[0]
    sf=(scaling_objective(sv+1e-6)-scaling_objective(sv-1e-6))/(2e-6)
    close(sg,sf,2e-6);assert torch.isfinite(sg)
    assert 0<max(r['max_alpha_change'] for r in scale_reports[1:4])<1e-3
    record('smooth_positive_utility_rescaling',scales=scale_reports,gradient_error=float((sg-sf).abs()))
    for name,pp,uu,mm in [
        ('one_token',psi[:,:1],u[:,:,:1],mask[:,:1]),
        ('identical_states',psi[:,:1].expand(-1,4,-1),u,mask),
        ('zero_utility',psi,torch.zeros_like(u),mask),
        ('constant_utility',psi,torch.full_like(u,.25),mask),
        ('near_zero_utility',psi,u*1e-12,mask)]:
        z=measured_attention(pp,phi,uu,mm);assert all(torch.isfinite(z[k]).all() for k in ['alpha','effect','aggregate'])
        rz,_=reference(pp.numpy(),phi.numpy(),uu.numpy(),mm.numpy());close(z['alpha'],torch.from_numpy(rz))
        record(name)
    # Actual source-generated states, not hand-entered substitute projectors.
    theta=torch.zeros((1,2,2,3,2),dtype=torch.float64);theta[0,0,0,0,0]=math.pi/2;theta[0,1,0,0,0]=-math.pi/2
    wp=circuit(theta);wq=circuit(torch.zeros((40,2,3,2),dtype=torch.float64))
    wu=torch.zeros((1,40,2),dtype=torch.float64);wu[0,1]=torch.tensor([.1,-.1],dtype=torch.float64);wu[0,2]=torch.tensor([1.,-1.],dtype=torch.float64)
    wm=torch.ones((1,2),dtype=torch.bool)
    full=measured_attention(wp,wq,wu,wm);pinch=measured_attention(wp,wq,wu,wm,kind='pinched');old=measured_attention(wp,wq,wu,wm,eta=0)
    close(pinch['alpha'],old['alpha']);assert float((full['alpha']-old['alpha']).abs().max())>1e-5
    bias=torch.full((40,),-10.,dtype=torch.float64);bias[1:3]=0
    target=torch.tensor([1]);ce0=F.cross_entropy(old['aggregate']+bias,target);ce1=F.cross_entropy(full['aggregate']+bias,target)
    assert ce1>ce0 and full['aggregate'][0,1]>0 and full['aggregate'][0,2]>full['aggregate'][0,1]
    record('source_circuit_offdiagonal_and_harmful_ce',old_ce=float(ce0),new_ce=float(ce1),full_pinched_alpha_max=float((full['alpha']-pinch['alpha']).abs().max()))
    # Strong non-quantum probability-space control, access matched not compute matched.
    center=wu-(old['alpha']*wu).sum(-1,keepdim=True);scale=center.abs().amax(-1,keepdim=True);scale=torch.where(scale>0,scale,torch.ones_like(scale))
    beta=old['alpha']*torch.exp(center/scale);beta=beta/beta.sum(-1,keepdim=True)
    assert float((beta-old['alpha']).abs().max())>1e-5
    record('simplex_control_also_changes_witness')
    permutation=utility_permutation(['fixed-a','fixed-b'],mask)
    expected=u.gather(-1,permutation[:,None].expand_as(u))
    intervention=measured_attention(psi,phi,u,mask,construction_utilities=expected)
    wrongread=measured_attention(psi,phi,expected,mask)
    close(intervention['alpha'],wrongread['alpha'])
    close(intervention['aggregate'],(intervention['alpha']*u*mask[:,None]).sum(-1))
    assert float((intervention['aggregate']-wrongread['aggregate']).abs().max())>1e-6
    assert torch.equal(permutation[~mask],torch.arange(4).expand(2,4)[~mask])
    record('permutation_changes_construction_only')
    # Independent digest-as-integer reconstruction, including a fixed identity fallback.
    expected_rows=[]
    for identity,row in zip(['fixed-a','fixed-b'],mask.tolist()):
        positions=[j for j,valid in enumerate(row) if valid]
        order=sorted(positions,key=lambda j:int(hashlib.sha256(('r93-utility-v2:'+identity+':'+str(j)).encode()).hexdigest(),16))
        if len(order)>1 and order==positions:order=order[1:]+order[:1]
        mapping=list(range(len(row)))
        for dst,src in zip(positions,order):mapping[dst]=src
        expected_rows.append(mapping)
    assert permutation.tolist()==expected_rows
    fallback_id='identity-fixture-1'
    assert sorted([0,1],key=lambda j:hashlib.sha256(f'r93-utility-v2:{fallback_id}:{j}'.encode()).digest())==[0,1]
    assert utility_permutation([fallback_id],torch.tensor([[True,True,False]])).tolist()==[[1,0,2]]
    assert utility_permutation(['single'],torch.tensor([[False,True,False]])).tolist()==[[0,1,2]]
    record('permutation_independent_mapping_fallback_and_singleton')
    # Derivatives through source circuit angles and utilities, at a smooth point.
    base_theta=torch.full((1,2,2,3,2),.14,dtype=torch.float64);dir_theta=torch.zeros_like(base_theta);dir_theta[0,0,0,0,0]=1
    base_query=torch.full((1,2,3,2),-.12,dtype=torch.float64);dir_query=torch.zeros_like(base_query);dir_query[0,0,0,0]=1
    def objective(x):
        ps=circuit(base_theta+x[0]*dir_theta);ph=circuit(base_query+x[1]*dir_query)
        util=torch.stack((x[2],-x[2])).reshape(1,1,2)
        return measured_attention(ps,ph,util,wm)['aggregate'].sum()
    x=torch.tensor([.23,-.11,.37],dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(objective(x),x)[0];fd=[]
    for j in range(3):
        delta=torch.zeros_like(x);delta[j]=1e-6;fd.append((objective(x+delta)-objective(x-delta))/(2e-6))
    fd=torch.stack(fd);close(grad,fd,2e-6)
    assert torch.isfinite(grad).all()
    for val in [0.,1e-12]:
        v=torch.full((1,1,2),val,dtype=torch.float64,requires_grad=True)
        z=measured_attention(wp,wq[:1],v,wm)['aggregate'].sum();g=torch.autograd.grad(z,v)[0];assert torch.isfinite(g).all()
    record('autograd_zero_nearzero_and_smooth_finite_difference',maximum_gradient_error=float((grad-fd).abs().max()))
    # Source and query circuit gradients at repeated eigenvalues / degenerate A.
    degenerate=[]
    for name,n,kind in [('one_token',1,'vary'),('identical_states',2,'vary'),('zero_utility',2,'zero'),('constant_utility',2,'constant'),('near_zero_utility',2,'near')]:
        identical=name=='identical_states'
        st=torch.zeros((1,n,2,3,2),dtype=torch.float64) if identical else theta[:,:n].clone()
        qt=torch.zeros((1,2,3,2),dtype=torch.float64)
        st.requires_grad_();qt.requires_grad_()
        if kind=='zero':vv=torch.zeros((1,1,n),dtype=torch.float64)
        elif kind=='constant':vv=torch.full((1,1,n),.25,dtype=torch.float64)
        else:vv=torch.linspace(-.3,.3,n,dtype=torch.float64).reshape(1,1,n)*(1e-12 if kind=='near' else 1.)
        vv.requires_grad_();dm=torch.ones((1,n),dtype=torch.bool)
        dz=measured_attention(circuit(st),circuit(qt),vv,dm)
        if name in ['one_token','identical_states','zero_utility','constant_utility']:assert float(dz['covariance'].abs().max())<2e-10
        dw=torch.arange(n,dtype=torch.float64).reshape(1,1,n)
        ew=torch.arange(64,dtype=torch.float64).reshape(8,8)/64
        objective=dz['aggregate'].sum()+(dz['alpha']*dw).sum()+(dz['effect'].real*ew).sum()
        gs=torch.autograd.grad(objective,(st,qt,vv))
        assert all(g is not None and torch.isfinite(g).all() for g in gs)
        degenerate.append(dict(case=name,max_covariance=float(dz['covariance'].detach().abs().max()),source_query_utility_gradient_maxima=[float(g.abs().max()) for g in gs]))
    record('degenerate_source_query_utility_gradients',cases=degenerate)
    torch.manual_seed(947)
    native=RMEAHead('quantum').double();new=UtilityMeasurementHead(eta=0).double();new.load_state_dict(native.state_dict())
    assert set(new.state_dict())==set(native.state_dict()) and sum(p.numel() for p in new.parameters())==81028
    h=torch.randn((1,3,1024),dtype=torch.float64);roles=torch.zeros((1,3,4),dtype=torch.float64);roles[0,0,2]=1;roles[0,1,3]=1;mm=torch.ones((1,3),dtype=torch.bool)
    zl,za=native(h,roles,mm);nl,na=new(h,roles,mm);close(zl,nl);close(za,na)
    new.update_eta=1.;loss=new(h,roles,mm)[0].square().sum();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    record('head_zero_equivalence_same_parameters_and_end_to_end_gradient',parameters=81028)
    rejected=0
    for args in [(psi,phi,u,torch.zeros_like(mask)),(psi,phi*2,u,mask),(psi.to(torch.complex64),phi,u,mask),(psi,phi,u[:,:,:2],mask)]:
        try:measured_attention(*args)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid input accepted')
    record('invalid_inputs_rejected',count=rejected)
    return dict(status='UTILITY_MEASUREMENT_NUMERICAL_CANARY_COMPLETE',checks=checks,
        check_count=len(checks),task_training=False,optimizer_steps=0,new_checkpoints=0,
        real_dataset_reads=0,COS_requests=0,formal_handoff=False,L5='NOT_ACCEPTED',
        qualification='Float64 CPU numerical canary only; no task utility, full Case Study or architecture completion',
        tolerances=dict(default_absolute=2e-10,gradient_absolute=2e-6,finite_difference_step=1e-6),
        seed=20261119)
