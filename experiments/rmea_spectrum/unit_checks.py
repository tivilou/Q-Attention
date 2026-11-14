"""Operator, gradient, masking and matched-control checks; no task-utility claim."""
import math
import numpy as np
import torch
from legacy.rmea import circuit
from spectrum import SpectrumHead,VARIANTS,circuit_columns,spectral_probability,INITIAL_SPECTRUM


def numpy_unitary(angles):
    state=np.eye(8,dtype=np.complex128)
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
            gate=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
            dense=np.kron(np.eye(2**wire),np.kron(gate,np.eye(2**(2-wire))))
            state=dense@state
        for control,target in ((0,1),(1,2),(2,0)):
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[perm]
    return state.T


def run_checks():
    torch.manual_seed(419);torch.set_num_threads(1);checks=[];worst=0.
    a=torch.randn(3,2,3,2,dtype=torch.float64)*.3;b=circuit_columns(a)
    np.testing.assert_allclose(b.detach().numpy(),np.stack([numpy_unitary(x) for x in a.numpy()]),atol=2e-14)
    torch.testing.assert_close(b[:,0],circuit(a),atol=2e-14,rtol=0)
    torch.testing.assert_close(b@b.conj().transpose(-2,-1),torch.eye(8,dtype=torch.complex128).expand(3,8,8),atol=2e-14,rtol=0)
    checks+=['all8columns independent dense NumPy parity','column0 legacy parity','unitarity']
    psi=circuit(torch.randn(2,4,2,3,2,dtype=torch.float64)*.2)
    lam=torch.randn(3,8,dtype=torch.float64).softmax(-1);p,c=spectral_probability(psi,b,lam)
    effect=torch.einsum('rk,rkd,rke->rde',lam,b,b.conj())
    direct=torch.einsum('btd,rde,bte->brt',psi.conj(),effect,psi).real
    torch.testing.assert_close(p,direct,atol=2e-14,rtol=0)
    torch.testing.assert_close(effect,effect.conj().transpose(-2,-1),atol=2e-14,rtol=0)
    torch.testing.assert_close(torch.linalg.eigvalsh(effect),lam.sort(-1).values,atol=2e-14,rtol=0)
    assert (p>=0).all() and (p<=1).all()
    pu,_=spectral_probability(psi,b,torch.full_like(lam,1/8))
    torch.testing.assert_close(pu,torch.full_like(pu,1/8),atol=2e-14,rtol=0)
    one=torch.zeros_like(lam);one[:,0]=1
    po,_=spectral_probability(psi,b,one)
    torch.testing.assert_close(po,torch.einsum('rd,btd->brt',b[:,0].conj(),psi).abs().square(),atol=2e-14,rtol=0)
    checks+=['PSD trace1 eigenvalues','direct dense effect parity','probability bounds','uniform-spectrum uniform attention','rank1 static limit']
    permutation=torch.tensor([3,0,7,1,2,6,5,4])
    pp,_=spectral_probability(psi,b[:,permutation],lam[:,permutation])
    torch.testing.assert_close(pp,p,atol=2e-14,rtol=0)
    checks.append('joint basis/spectrum permutation invariance')
    # Same original channel0 p=0, but different OTHER channels: no scalar g(p0)
    # can reproduce both. This is an expressivity witness, not task advantage.
    basis=torch.eye(8,dtype=torch.complex128)[None];words=basis[:,[1,2],:]
    weights=torch.tensor([[.1,.6,.2,.02,.02,.02,.02,.02]],dtype=torch.float64)
    witness,channels=spectral_probability(words,basis,weights)
    assert channels[0,0,0,0]==channels[0,0,1,0]==0
    assert abs(float(witness[0,0,0])-.6)<1e-14 and abs(float(witness[0,0,1])-.2)<1e-14
    checks.append('equal original score/distinct spectral score witness')
    wa=(torch.randn(1,2,2,3,2,dtype=torch.float64)*.3).requires_grad_()
    ea=(torch.randn(2,2,3,2,dtype=torch.float64)*.3).requires_grad_()
    beta=torch.randn(2,8,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda w,e,l:spectral_probability(circuit(w),circuit_columns(e),l.softmax(-1))[0],(wa,ea,beta),eps=1e-6,atol=3e-5,rtol=3e-4)
    checks.append('word/effect/spectrum finite-difference gradcheck')
    h=torch.randn(2,4,1024,dtype=torch.float64)*.03;roles=torch.randn(2,4,4,dtype=torch.float64)*.1
    mask=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool);models={};counts={}
    for variant in VARIANTS:
        torch.manual_seed(419);head=SpectrumHead(variant).double();models[variant]=head
        counts[variant]=sum(x.numel() for x in head.parameters())
        d=head(h,roles,mask,details=True)
        torch.testing.assert_close(d['alpha'].sum(-1),torch.ones(2,40,dtype=torch.float64),atol=2e-12,rtol=0)
        assert not d['alpha'][1,:,2:].any() and torch.isfinite(d['logits']).all()
        perturbed=h.clone();perturbed[1,2:]=123
        torch.testing.assert_close(head(perturbed,roles,mask)[0],d['logits'],atol=2e-12,rtol=0)
        loss=d['logits'].square().mean();loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in head.parameters())
        if variant!='static':
            g=head.spectrum_logits.grad
            assert float(g.abs().max())>1e-10
            torch.testing.assert_close(g.sum(-1),torch.zeros(40,dtype=torch.float64),atol=1e-12,rtol=0)
    assert counts['static']==81028 and all(counts[k]==81348 for k in VARIANTS[1:])
    for name in ('value.weight','value.bias','decoder','bias','prepare.weight','prepare.bias','effect'):
        for head in models.values():torch.testing.assert_close(dict(head.named_parameters())[name],dict(models['static'].named_parameters())[name],atol=0,rtol=0)
    s=models['spectral'];c=models['scalar_calibration']
    for key in ('probability','alpha','logits'):
        ds=s(h,roles,mask,details=True)[key];dc=c(h,roles,mask,details=True)[key]
        worst=max(worst,float((ds-dc).abs().max()));torch.testing.assert_close(ds,dc,atol=5e-8,rtol=2e-7)
    for name in ('value.weight','value.bias','decoder','bias','prepare.weight','prepare.bias','effect'):
        torch.testing.assert_close(dict(s.named_parameters())[name].grad,dict(c.named_parameters())[name].grad,atol=3e-8,rtol=3e-6)
    checks+=['all mode padding/normalization/finite gradients','all common initialization exact','spectral/scalar parameter parity','matched initial function and shared-parameter gradients','softmax gauge zero-sum gradient']
    return dict(status='PASS',checks=checks,parameter_counts=counts,matched_initialization_max_error=worst,
        added_nominal_parameters=320,added_spectrum_effective_dof=280,
        limitations=['Operator/property evidence only','Different functional classes despite matched parameter count','No task utility, training convergence, novelty or L3 claim'])
