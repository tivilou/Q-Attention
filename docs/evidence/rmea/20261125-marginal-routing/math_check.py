"""Independent local-overlap and fixed-readout identities; no task data."""
from pathlib import Path
import json
import numpy as np

def reduced_direct(vector,dimensions,view):
    t=np.asarray(vector).reshape(dimensions);a,b,c=dimensions
    if view==0:
        return np.array([[sum(t[i,j,k]*t[ii,jj,k].conjugate() for k in range(c)) for ii in range(a) for jj in range(b)] for i in range(a) for j in range(b)])
    return np.array([[sum(t[i,j,k]*t[i,jj,kk].conjugate() for i in range(a)) for jj in range(b) for kk in range(c)] for j in range(b) for k in range(c)])

def direct_probabilities(states,queries,dimensions,population=False):
    return np.array([[[sum((np.diag(reduced_direct(q,dimensions,k)).real*np.diag(reduced_direct(s,dimensions,k)).real)) if population else np.trace(reduced_direct(q,dimensions,k)@reduced_direct(s,dimensions,k)).real for k in range(2)] for s in states] for q in queries])

def main():
    rng=np.random.default_rng(20261125);errors=[];identities=[];eigen=[]
    for dims in ((2,2,2),(2,3,2)):
        d=int(np.prod(dims))
        for i in range(32):
            s=rng.normal(size=d)+1j*rng.normal(size=d);s/=np.linalg.norm(s)
            q=rng.normal(size=d)+1j*rng.normal(size=d);q/=np.linalg.norm(q)
            for view in range(2):
                rho=reduced_direct(s,dims,view);sigma=reduced_direct(q,dims,view)
                effect=np.kron(sigma,np.eye(dims[2])) if view==0 else np.kron(np.eye(dims[0]),sigma)
                errors.append(abs(np.trace(sigma@rho)-s.conj()@effect@s));eigen.append(float(np.linalg.eigvalsh(rho).min()))
            alpha=rng.dirichlet(np.ones(5),size=(40,2)).transpose(0,2,1)
            utility=rng.normal(size=(40,5,2))
            separate=(alpha*utility).sum((1,2));tied=(alpha.mean(-1,keepdims=True)*utility).sum((1,2))
            delta=.5*((alpha[...,0]-alpha[...,1])*(utility[...,0]-utility[...,1])).sum(-1)
            identities.append(float(np.max(abs(separate-tied-delta))))
    q=np.eye(8)[0:1];s=np.eye(8)[[1,4]];p=direct_probabilities(s,q,(2,2,2))[0]
    assert np.array_equal(p,np.array([[1.,0.],[0.,1.]]))
    alpha=(p+1e-6)/(p+1e-6).sum(0,keepdims=True)
    split=float((alpha*np.eye(2)).sum());assert abs(split-2*(1+1e-6)/(1+2e-6))<1e-14 and split>1
    plus=(np.eye(8)[0]+np.eye(8)[7])/np.sqrt(2);minus=(np.eye(8)[0]-np.eye(8)[7])/np.sqrt(2)
    assert all(np.allclose(reduced_direct(plus,(2,2,2),k),reduced_direct(minus,(2,2,2),k)) for k in range(2))
    assert abs(np.vdot(plus,minus))<1e-14
    assert max(errors)<1e-12 and max(identities)<1e-12 and min(eigen)>-1e-12
    out=dict(status='PASS',random_pairs=64,full_effect_error=float(max(errors)),readout_identity_error=max(identities),minimum_reduced_eigenvalue=min(eigen),single_route_witness_max=1.,separate_route_witness=split,ghz_sign_information_lost=True,scope='Known partial trace/grouped pooling; no text utility or T acceptance')
    (Path(__file__).parent/'math-check.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))

if __name__=='__main__':main()
