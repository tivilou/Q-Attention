"""Small label-free operator/property fixtures, run inside the registered job."""
import numpy as np
from contrast import branch_weights


def check():
    rng=np.random.default_rng(20261114)
    checks=[]
    worst=0.
    for n in (1,2,7,23):
        for repeat in range(10):
            p=rng.uniform(0,1,(2,4,n));u=rng.normal(size=p.shape)
            mask=np.ones((2,n),bool)
            if n>1:mask[1,-1]=False
            a,b=branch_weights(p,mask);w=a-b
            q=(p+1e-6)/(1+2e-6);t=mask.sum(-1)[:,None]
            qm=(q*mask[:,None,:]).sum(-1)/t
            um=(u*mask[:,None,:]).sum(-1)/t
            cov=((q-qm[...,None])*(u-um[...,None])*mask[:,None,:]).sum(-1)/t
            direct=(w*u).sum(-1);independent=cov/(qm*(1-qm))
            np.testing.assert_allclose(direct,independent,atol=2e-10)
            worst=max(worst,float(np.abs(direct-independent).max()))
            np.testing.assert_allclose((w*(u+17)).sum(-1),direct,atol=2e-12)
            np.testing.assert_allclose(w.sum(-1),0,atol=1e-12)
            assert (np.abs(w).sum(-1)<=2+1e-12).all()
            hi=np.where(mask[:,None,:],u,-np.inf).max(-1)
            lo=np.where(mask[:,None,:],u,np.inf).min(-1)
            assert (np.abs(direct)<=hi-lo+1e-12).all()
            ar,br=branch_weights(np.repeat(p,2,axis=-1),np.repeat(mask,2,axis=-1))
            np.testing.assert_allclose(((ar-br)*np.repeat(u,2,axis=-1)).sum(-1),direct,atol=1e-12)
            for constant in (0.,.25,1.):
                ac,bc=branch_weights(np.full_like(p,constant),mask)
                np.testing.assert_allclose(ac-bc,0,atol=1e-12)
            checks.append(f'n{n}-fixture{repeat}')
    # Explicit loss of common utility and T=1 information; these are limitations.
    a,b=branch_weights(np.array([[[.1,.9]]]),np.array([[1,1]],bool))
    assert float(((a-b)*np.array([[[0.,1.]]])).sum())>.79
    for bad in (-.1,1.1):
        try:branch_weights(np.full((1,1,1),bad),np.ones((1,1),bool))
        except ValueError:pass
        else:raise AssertionError('Invalid probability accepted')
    return dict(status='PASS',fixtures=len(checks),max_covariance_identity_error=worst,
        properties=['binary branches normalized','signed mass zero','padding excluded',
            'utility offset invariant','replication invariant','range bound',
            'constant probability and one-token collapse','invalid probabilities rejected'],
        scope='Algebra/property checks only; no task utility or quantum advantage')
