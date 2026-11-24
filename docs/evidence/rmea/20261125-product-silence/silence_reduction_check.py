"""Scientific algebra check: dense product POVM versus classical silence likelihood."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parent
rng=np.random.default_rng(20261125)
def unit(z):return z/np.linalg.norm(z,axis=-1,keepdims=True)
def detector(q):
    eps=1e-6
    return np.eye(8)-(q.T@q.conj()+39*eps*np.eye(8))/(39*(1+eps))
max_error=0.;min_eig=1.;max_eig=0.;permutation_error=0.
for trial in range(128):
    q=unit(rng.normal(size=(39,8))+1j*rng.normal(size=(39,8)))
    psi=unit(rng.normal(size=(2,8))+1j*rng.normal(size=(2,8)))
    f=detector(q);e=np.linalg.eigvalsh(f)
    min_eig=min(min_eig,float(e.min()));max_eig=max(max_eig,float(e.max()))
    p=(abs(q.conj()@psi.T)**2+1e-6).mean(0)/(1+1e-6)
    state=np.kron(psi[0],psi[1])
    dense=np.vdot(state,np.kron(f,f)@state)
    scalar=(1-p[0])*(1-p[1])
    max_error=max(max_error,abs(dense-scalar))
    reverse=np.kron(psi[1],psi[0])
    permutation_error=max(permutation_error,abs(dense-np.vdot(reverse,np.kron(f,f)@reverse)))
    # Mask drops the second tensor factor, not the whole input-conditioned state.
    max_error=max(max_error,abs(np.vdot(psi[0],f@psi[0])-(1-p[0])))
assert max_error<1e-12 and permutation_error<1e-12 and min_eig>=-1e-12 and max_eig<=1+1e-12
# Both token states depend on one common classical input. Conditional tensor
# factorization remains exact; test the same trainable scalar derivative.
q=unit(rng.normal(size=(39,8))+1j*rng.normal(size=(39,8)));f=detector(q)
a=rng.normal(size=(2,8))+1j*rng.normal(size=(2,8));b=rng.normal(size=(2,8))+1j*rng.normal(size=(2,8))
def value(theta,dense):
    psi=unit(a+theta*b)
    if dense:
        x=np.kron(psi[0],psi[1]);return float(np.vdot(x,np.kron(f,f)@x).real)
    p=(abs(q.conj()@psi.T)**2+1e-6).mean(0)/(1+1e-6)
    return float(np.prod(1-p))
h=1e-5
derivatives=[(value(.2+h,d)-value(.2-h,d))/(2*h) for d in (True,False)]
assert abs(derivatives[0]-derivatives[1])<1e-9
record=dict(status='PASS',fixtures=128,dimensions=8,positive_relations=39,
    dense_scalar_max_error=float(max_error),permutation_error=float(permutation_error),
    silence_effect_min_eigenvalue=min_eig,silence_effect_max_eigenvalue=max_eig,
    common_context_parameter_derivatives=derivatives,derivative_error=abs(derivatives[0]-derivatives[1]),
    new_model_fits=0,new_dataset_reads=0,claim='Identity for product silence effects on product token preparations; not all joint effects or preparations, no utility or novelty evidence')
(r/'silence-reduction-check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
