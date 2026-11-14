"""Gauge-invariant descriptive geometry; no task inputs or model fitting."""
import numpy as np

def dense_states(angles):
    result=[]
    for a in angles:
        v=np.zeros(8,dtype=np.complex128);v[0]=1
        for layer in range(2):
            for wire in range(3):
                y,z=a[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
                gate=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
                v=np.kron(np.eye(2**wire),np.kron(gate,np.eye(2**(2-wire))))@v
            for control,target in ((0,1),(1,2),(2,0)):
                gate=np.zeros((8,8))
                for j in range(8):gate[j^((1<<(2-target)) if (j>>(2-control))&1 else 0),j]=1
                v=gate@v
        result.append(v)
    return np.stack(result)

def describe(states):
    a=np.asarray(states,dtype=np.complex128)
    assert a.ndim==2 and np.isfinite(a).all()
    r,d=a.shape;assert r>d>1
    np.testing.assert_allclose(np.sum(np.abs(a)**2,axis=1),1,atol=2e-12,rtol=0)
    sigma=a.T@a.conj()/r
    eig=np.linalg.eigvalsh(sigma)
    assert eig.min()>-2e-12
    eig=np.maximum(eig,0);np.testing.assert_allclose(eig.sum(),1,atol=2e-12,rtol=0)
    gram=np.abs(a.conj()@a.T)**2
    purity=float(np.sum(eig**2))
    np.testing.assert_allclose(purity,np.sum(np.abs(sigma)**2),atol=2e-12,rtol=0)
    np.testing.assert_allclose(purity,gram.sum()/r**2,atol=2e-12,rtol=0)
    off=gram[np.triu_indices(r,1)];mean=float(off.mean())
    np.testing.assert_allclose(mean,(r*purity-1)/(r-1),atol=2e-12,rtol=0)
    positive=eig[eig>0]
    return dict(relations=r,dimension=d,eigenvalues=eig.tolist(),purity=purity,
        inverse_purity_effective_rank=1/purity,entropy_effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))),
        normalized_frame_excess=d*purity-1,welch_lower_bound=(r-d)/(d*(r-1)),
        offdiag_mean_overlap_squared=mean,offdiag_quantiles=np.quantile(off,[0,.25,.5,.75,.95,1]).tolist(),
        unordered_pairs=len(off),pairs_overlap_gt_point9=int((off>.9).sum()),pairs_overlap_gt_point99=int((off>.99).sum()),
        purity_trace_gram_identity='PASS')
