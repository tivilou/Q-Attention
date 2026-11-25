"""Known mean-state unitary flow: scientific algebra checks only, no task rows."""
from pathlib import Path
import json, math, hashlib
import numpy as np
def expm(A):
    # Every input here is skew-Hermitian. No SciPy installation is needed.
    H=1j*A
    assert np.max(np.abs(H-H.conj().T))<1e-12
    lam,V=np.linalg.eigh(H)
    return (V*np.exp(-1j*lam))@V.conj().T
EPS=1e-6
rng=np.random.default_rng(20261125)
errors={k:0. for k in ('dual','mass','readout','derivative','t0','gauge','duplicate','bound')}
changes=[]
for case in range(96):
    d=8;n=2+case%15
    X=rng.normal(size=(d,n))+1j*rng.normal(size=(d,n))
    X/=np.linalg.norm(X,axis=0,keepdims=True)
    q=rng.normal(size=d)+1j*rng.normal(size=d);q/=np.linalg.norm(q)
    u=rng.normal(size=n)
    S=X@X.conj().T/n;G=X.conj().T@X/n
    V=(X*u)@X.conj().T
    def flow(t,X=X,S=S,q=q,u=u):
        w=expm(-1j*t*S)@q
        raw=np.abs(X.conj().T@w)**2+EPS
        return raw/raw.sum(),raw.sum(),float(raw@u/raw.sum())
    a,D,z=flow(math.pi);a0,D0,z0=flow(0)
    v=X.conj().T@q
    errors['dual']=max(errors['dual'],float(np.max(np.abs(expm(-1j*math.pi*G)@v-X.conj().T@expm(-1j*math.pi*S)@q))))
    errors['mass']=max(errors['mass'],abs(D-D0))
    exact=float((np.vdot(q,expm(1j*math.pi*S)@V@expm(-1j*math.pi*S)@q).real+EPS*u.sum())/D0)
    errors['readout']=max(errors['readout'],abs(exact-z))
    h=1e-5
    fd=(flow(h)[2]-flow(-h)[2])/(2*h)
    derivative=float((1j*np.vdot(q,(S@V-V@S)@q)).real/D0)
    errors['derivative']=max(errors['derivative'],abs(fd-derivative))
    native=(abs(v)**2+EPS);native/=native.sum()
    errors['t0']=max(errors['t0'],float(np.max(abs(a0-native))))
    phases=np.exp(1j*rng.uniform(-math.pi,math.pi,n));Xg=X*phases
    ag,_,_=flow(math.pi,X=Xg)
    errors['gauge']=max(errors['gauge'],float(np.max(abs(a-ag))))
    X2=np.repeat(X,2,axis=1);u2=np.repeat(u,2)
    ad,_,zd=flow(math.pi,X=X2,S=X2@X2.conj().T/(2*n),u=u2)
    errors['duplicate']=max(errors['duplicate'],float(np.max(abs(ad.reshape(n,2).sum(1)-a))),abs(zd-z))
    B=math.pi*np.linalg.norm(S@V-V@S,ord=2)/D0
    errors['bound']=max(errors['bound'],max(abs(z-z0)-B,0.))
    changes.append(z-z0)
# Degenerate mean state is an exact global phase, not a changed measurement.
X=np.eye(8,dtype=complex);q=np.ones(8,dtype=complex)/np.sqrt(8)
assert np.max(abs(abs(X.conj().T@expm(-1j*math.pi*np.eye(8)/8)@q)**2-abs(X.conj().T@q)**2))<1e-14
assert max(errors.values())<1e-8
assert min(changes)<0<max(changes)
r=dict(status='PASS_ALGEBRA_ONLY',cases=96,dimension=8,time=math.pi,EPS=EPS,max_errors=errors,
       min_signed_utility_change=min(changes),max_signed_utility_change=max(changes),
       known_tools='Matrix-exponential intertwining, unitary invariance, Ehrenfest commutator and norm bounds',
       claim_scope='Synthetic state-port only; no task efficacy, parameter reachability, training gradients, quantum exclusivity or novel theory',
       task_rows=0,new_fits=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path(__file__).with_name('algebra-verification.json').write_bytes((json.dumps(r,indent=2)+'\n').encode())
print(json.dumps(r))
