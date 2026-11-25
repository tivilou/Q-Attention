from pathlib import Path
import json
import numpy as np

r=Path(__file__).resolve().parent
cfg=json.loads((r/'scope.json').read_bytes())
eps,d=cfg['EPS'],cfg['dimension']
e=np.eye(d);plus=(e[0]+e[1])/np.sqrt(2);minus=(e[0]-e[1])/np.sqrt(2);phi=e[0].astype(complex)
values=np.zeros((2,64));values[:,0]=[-1,1];values[:,1]=[2,-3];u=values[:,0]


def exp_apply(m,v,coefficient):
    lam,vec=np.linalg.eigh(m)
    return vec@(np.exp(coefficient*lam)*(vec.conj().T@v))


def calculate(states):
    rho=np.einsum('ti,tj->tij',states,states.conj())
    bts=rho+eps*np.eye(d);b=bts.sum(0);a=np.einsum('t,tij->ij',u,bts)
    def read(q): return float(np.vdot(q,a@q).real/np.vdot(q,b@q).real)
    raw=abs(states.conj()@phi)**2+eps;alpha=raw/raw.sum()
    original=read(phi);c=a-original*b
    q=exp_apply(c,phi,1.)
    qr=exp_apply(c.real,phi.real,1.)
    guided=read(q);real8=read(qr)
    flow=read(exp_apply(rho.mean(0),phi,-1j*np.pi))
    lam,vec=np.linalg.eigh(b)
    whiten=(vec*(lam**-.5))@vec.conj().T
    frame=read(whiten@phi)
    p=(abs(states)**2)@(abs(phi)**2)+eps;da=p/p.sum()
    tau=1.;tilt=alpha*np.exp(tau*u);tilt/=tilt.sum()
    amplitude=states@phi.conj();ss=amplitude.sum();vv=(amplitude*u).sum()
    weak=float(((ss.conjugate()*vv).real+eps*u.sum())/(abs(ss)**2+2*eps))
    mean=alpha@values;product=mean[0::2]*mean[1::2]
    covariance=(alpha[:,None]*(values[:,0::2]*values[:,1::2])).sum(0)-product
    distinct=product-(alpha[:,None]**2*(values[:,0::2]*values[:,1::2])).sum(0)
    collision=(1-(alpha**2).sum())*product
    return dict(alpha=alpha.tolist(),population=abs(states).tolist(),original=original,guided=guided,real8=real8,
                gen90flow=flow,wordframe=frame,dephased=float(da@u),scalar_tilt=float(tilt@u),weak=weak,
                gen97_covariance=covariance.tolist(),gen97_meanproduct=product.tolist(),
                gen98_distinct=distinct.tolist(),gen98_collision=collision.tolist())


aa=np.stack([plus,plus]).astype(complex);bb=np.stack([plus,minus]).astype(complex)
a,b=calculate(aa),calculate(bb)
expected=np.tanh(2)/(1+2*eps)
assert abs(a['guided'])<1e-14 and abs(b['guided']-expected)<1e-14
for key in ('alpha','population','original','dephased','scalar_tilt','gen90flow','wordframe','weak',
            'gen97_covariance','gen97_meanproduct','gen98_distinct','gen98_collision'):
    assert np.max(abs(np.array(a[key])-np.array(b[key])))<1e-12,key
assert abs(b['real8']-b['guided'])<1e-14
phased=calculate(bb*np.exp(1j*np.array([.2,.7]))[:,None])
assert abs(phased['guided']-b['guided'])<1e-14
# Populations are explicitly squared below; comparison above used absolute amplitudes,
# which are also identical and do not assert a new primitive.
for result,states in ((a,aa),(b,bb)):
    result['population']=(abs(states)**2).tolist()
out=dict(status='PASS_RESIDUAL_GEOMETRY_FUNCTIONAL_CONTRAST',context_A=a,context_B=b,
         predicted_difference=float(expected),real8_exact_for_this_real_fixture=True,
         independent_word_global_phase_invariance_error=abs(phased['guided']-b['guided']),
         native_ansatz_joint_reachability=False,task_utility=False,literature_novelty=False,
         selected_task_method=False,new_theory=False,external_research_review=False,
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'contrast-results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in out.items() if k not in ('context_A','context_B')}))
