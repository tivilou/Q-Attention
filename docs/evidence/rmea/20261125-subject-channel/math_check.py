"""Independent NumPy reference: average transported pure-source probabilities."""
from pathlib import Path
import json
import numpy as np

def numpy_scores(states,unitaries,subject,population=False):
    answer=[]
    for u in unitaries:
        transformed=(u@states[subject].T).T
        if population:p=np.mean(abs(transformed)**2,axis=0)@abs(states.T)**2
        else:p=np.mean(abs(transformed.conj()@states.T)**2,axis=0)
        answer.append(p)
    return np.array(answer)

def run():
    rng=np.random.default_rng(20261125);maximum=0.;minimum=1.;isotropic=0.
    for _ in range(64):
        state=rng.normal(size=(7,8))+1j*rng.normal(size=(7,8));state/=np.linalg.norm(state,axis=-1,keepdims=True)
        u=np.stack([np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))[0] for _ in range(4)])
        source=np.array([True,False,False,True,False,False,False])
        rho=state[source].T@state[source].conj()/2
        e=np.einsum('rik,kl,rjl->rij',u,rho,u.conj())
        direct=np.einsum('ti,rij,tj->rt',state.conj(),e,state).real
        maximum=max(maximum,float(abs(direct-numpy_scores(state,u,source)).max()))
        population=np.einsum('ri,ti->rt',np.diagonal(e,axis1=-2,axis2=-1).real,abs(state)**2)
        maximum=max(maximum,float(abs(population-numpy_scores(state,u,source,True)).max()))
        minimum=min(minimum,float(np.linalg.eigvalsh(e).min()))
        isotropic=max(isotropic,float(abs(u@(np.eye(8)/8)@u.conj().transpose(0,2,1)-np.eye(8)/8).max()))
        phase=np.exp(1j*rng.normal(size=7));maximum=max(maximum,float(abs(numpy_scores(state*phase[:,None],u,source)-direct).max()))
    assert maximum<1e-12 and minimum>-1e-12 and isotropic<1e-12
    # Static-source collapse is an adverse example, not a utility guarantee.
    rec=dict(status='PASS',fixtures=64,source_average_dense_error=maximum,effect_min_eigenvalue=minimum,isotropic_uniform_error=isotropic,new_model_fits=0,scope='Known unitary-conjugated density algebra, not newT or task evidence')
    Path(__file__).with_name('math-check.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode());print(json.dumps(rec))
if __name__=='__main__':run()
