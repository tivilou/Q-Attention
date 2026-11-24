"""Known spectral upper witnesses for two shared-measurement constraints."""
from pathlib import Path
import json
import numpy as np

TOL=1e-7

def normalize(c):
    norm=np.linalg.norm(c,axis=(-1,-2))
    return c/np.maximum(norm,1e-300)[...,None,None]

def upper(a,b,weight):
    matrix=weight[...,None,None]*a+(1-weight[...,None,None])*b
    return np.linalg.eigvalsh(matrix)[...,-1]

def dual_search(a,b):
    assert a.shape==b.shape and a.ndim==3
    lo=np.zeros(len(a));hi=np.ones(len(a));ratio=(np.sqrt(5.)-1.)/2
    x=hi-ratio*(hi-lo);y=lo+ratio*(hi-lo)
    fx=upper(a,b,x);fy=upper(a,b,y)
    best_weight=np.full(len(a),.5);best_value=upper(a,b,best_weight)
    for _ in range(48):
        for w,v in ((x,fx),(y,fy)):
            better=v<best_value;best_weight=np.where(better,w,best_weight);best_value=np.minimum(best_value,v)
        left=fx<=fy
        hi=np.where(left,y,hi);lo=np.where(left,lo,x)
        x=hi-ratio*(hi-lo);y=lo+ratio*(hi-lo)
        fx=upper(a,b,x);fy=upper(a,b,y)
    weights=np.stack((np.zeros(len(a)),np.ones(len(a)),best_weight,x,y),-1)
    vals=np.stack([upper(a,b,weights[:,k]) for k in range(5)],-1)
    pick=vals.argmin(-1);rows=np.arange(len(a))
    return weights[rows,pick],vals[rows,pick]

def fixtures():
    rng=np.random.default_rng(20261125);x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);eye=np.eye(2)
    # All individual maxima are positive; neither complementary nor X/Z pair can share a solution.
    a=np.stack((np.diag([.25,-.75]),x-.8*eye,np.diag([.5,-.5])))
    b=np.stack((np.diag([-.75,.25]),z-.8*eye,np.diag([.2,-.8])))
    weights,values=dual_search(a,b)
    expected=np.array([-.25,1/np.sqrt(2)-.8,.2])
    err=float(abs(values-expected).max());assert err<1e-10
    assert np.all(np.linalg.eigvalsh(a)[:,-1]>0) and np.all(np.linalg.eigvalsh(b)[:,-1]>0)
    assert values[0]<0 and values[1]<0 and values[2]>0
    # A fixed query satisfies the third pair; sample arbitrary density matrices to check the upper inequality.
    worst=0.
    for d in (2,8,12):
        for complex_input in (False,True):
            c=rng.normal(size=(7,2,d,d))
            if complex_input:c=c+1j*rng.normal(size=c.shape)
            c=(c+c.conj().swapaxes(-1,-2))/2
            aa,bb=normalize(c[:,0]),normalize(c[:,1]);ww,uu=dual_search(aa,bb)
            for _ in range(20):
                v=rng.normal(size=(7,d))
                if complex_input:v=v+1j*rng.normal(size=v.shape)
                v/=np.linalg.norm(v,axis=-1,keepdims=True)
                av=np.einsum('bi,bij,bj->b',v.conj(),aa,v).real
                bv=np.einsum('bi,bij,bj->b',v.conj(),bb,v).real
                worst=max(worst,float((np.minimum(av,bv)-uu).max()))
            assert np.min(ww)>=0 and np.max(ww)<=1
    assert worst<1e-12
    return dict(status='PASS',analytical_pair_upper_bounds=values.tolist(),analytical_max_error=err,random_pure_upper_violation=max(0.,worst),mixed_effect_extension='Linearity and PSD trace-one support identity; pure samples are fixtures, not proof',claim='Known support-function mathematics; no new T or task result')

if __name__=='__main__':
    v=fixtures();Path(__file__).with_name('math-check.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
