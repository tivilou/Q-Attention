"""Artificial d=8 reset equilibrium, resolvent adjoint and sensitivity tests."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
import numpy as np

D=8
EPS=1e-6
I=np.eye(D)

def herm(a):return (a+a.conj().T)/2
def projector(a):return np.outer(a,a.conj())/np.vdot(a,a).real
def comm(a,b):return a@b-b@a
def norm(a):return float(np.linalg.norm(a))
def err(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def adjoint_pair(h,gamma):
    levels,u=np.linalg.eigh(h)
    f=gamma/(gamma+1j*(levels[:,None]-levels[None,:]))
    def apply(a,adj=False):return u@((f.conj() if adj else f)*(u.conj().T@a@u))@u.conj().T
    return apply,levels,u

def dense_matrix(h,gamma):
    # Independently assemble inverse system by acting on each matrix unit.
    units=np.eye(D*D).reshape(D*D,D,D)
    return np.column_stack([(gamma*x+1j*comm(h,x)).ravel() for x in units])

def dense_apply(a,gamma,mat):return np.linalg.solve(mat,gamma*a.ravel()).reshape(D,D)
def pooled(g,values,mask,w):
    mass=np.where(mask[:,None],g+EPS,0.)
    f=(mass.T@values)/mass.sum(axis=0)[:,None]
    return f,np.einsum('rk,rk->r',w,f)

def test():
    rng=np.random.default_rng(20261127)
    def z(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    u,_=np.linalg.qr(z((D,D)))
    # Repeated eigenvalues are deliberate; no appeal to eigenvector derivatives.
    h=u@np.diag([-.36,-.21,-.21,0.,0.,.12,.20,.36])@u.conj().T
    gamma=.75
    a=np.stack([projector(v) for v in z((7,D))]); e=np.stack([projector(v) for v in z((40,D))])
    mask=np.array([1,1,0,1,1,0,1],bool)
    v=rng.normal(size=(7,64)); w=rng.normal(size=(40,64))
    sigma=a[mask].mean(axis=0); apply,levels,energy_basis=adjoint_pair(h,gamma)
    m=dense_matrix(h,gamma); forward=np.stack([dense_apply(x,gamma,m) for x in a]); rho=dense_apply(sigma,gamma,m)
    eprime=np.stack([apply(ef,adj=True) for ef in e]);
    p0=np.einsum('rij,tji->tr',e,forward).real
    p1=np.einsum('rij,tji->tr',eprime,a).real
    # Conditional labels of a Poisson reset average also integrate over reset age.
    nodes,weights=np.polynomial.laguerre.laggauss(32)
    average=np.zeros_like(rho)
    for x,b in zip(nodes,weights):
        U=energy_basis@np.diag(np.exp(-1j*levels*x/gamma))@energy_basis.conj().T
        average+=b*U@sigma@U.conj().T
    f0,y0=pooled(p0,v,mask,w);f1,y1=pooled(p1,v,mask,w)
    newv=v.copy();newv[~mask]=2e6
    evals=np.linalg.eigvalsh(eprime)
    errors=dict(dense_vs_spectral=err(rho,apply(sigma)),stationary=err(gamma*(sigma-rho)-1j*comm(h,rho),0),
          external_quadrature=err(average,rho),source_mixture=err(forward[mask].mean(axis=0),rho),
          all_masses=err(p0,p1),all64_pool=err(f0,f1),all40_scores=err(y0,y1),padding=err(pooled(p1,newv,mask,w)[1],y1),
          trace=err(np.trace(forward,axis1=1,axis2=2),1.),effect_trace=err(np.trace(eprime,axis1=1,axis2=2),1.))
    assert max(errors.values())<3e-12 and evals.min()>-1e-13 and evals.max()<1+1e-13

    # Local diagnostic derivative, with a separate central difference at repeated H eigenvalues.
    dh=herm(z((D,D)))*.1
    s0=I/D
    ds=(a[0]-a[1])*.1
    rho0=dense_apply(s0+.2*(sigma-s0),gamma,m)
    state0=s0+.2*(sigma-s0)
    local=np.linalg.solve(m,(gamma*ds-1j*comm(dh,rho0)).ravel()).reshape(D,D)
    delta=1e-5
    plus=dense_apply(state0+delta*ds,gamma,dense_matrix(h+delta*dh,gamma))
    minus=dense_apply(state0-delta*ds,gamma,dense_matrix(h-delta*dh,gamma))
    slope_error=err(local,(plus-minus)/(2*delta)); assert slope_error<1e-9
    lip=norm(ds)+(2/gamma)*np.linalg.norm(dh,2)*norm(rho0)
    assert norm(local)<=lip and abs(np.trace(local))<1e-13

    # Rational sign witness and rank mutation, embedded in exactly the same d=8 interface.
    # E projects onto Y+, while H drives X between indices 0 and 4.
    x=np.zeros((D,D));x[0,4]=x[4,0]=.5
    state=np.stack([projector(I[0]),projector(I[4])]); term=projector(I[0]+1j*I[4]);
    signed=[];eff=[];mean=[]
    for sign in (1,-1):
        change=dense_matrix(sign*x,1.)
        ss=np.stack([dense_apply(j,1.,change) for j in state])
        signed.append([float(np.trace(term @ q).real) for q in ss])
        eff.append(adjoint_pair(sign*x,1.)[0](term,adj=True))
        mean.append(ss.mean(axis=0))
    exact=[[F(1,4),F(3,4)],[F(3,4),F(1,4)]]
    assert err(signed,np.asarray(exact,float))<1e-14 and err(mean[0],mean[1])<1e-14
    assert all(np.linalg.matrix_rank(q,tol=1e-12)==2 for q in eff)
    # Independent classical real-quadrature replay of the *witness*, not a trained comparator.
    # Dynamics dy = -omega z - gamma y, dz = omega y + gamma(z0-z).
    classical=[]
    for omega in (F(1),F(-1)):
        classical.append([(1-omega*sign/(1+omega**2))/2 for sign in (F(1),F(-1))])
    assert classical==exact
    # Full-energy dephasing preserves degenerate blocks; here it removes the Y effect.
    ew,ev=np.linalg.eigh(x); assert len(np.unique(np.round(ew,12)))==3
    energy_mask=np.isclose(ew[:,None]-ew[None,:],0.,atol=1e-14)
    energy_d=[float(np.trace(term @ (ev@((ev.conj().T@q@ev)*energy_mask)@ev.conj().T)).real) for q in state]
    assert err(energy_d,[.5,.5])<1e-14
    wrong_sign=err(signed,signed[::-1]); assert abs(wrong_sign-.5)<1e-14
    return dict(status='PASS_RESET_RESOLVENT_READOUT_AND_SENSITIVITY',errors=errors,
      derivative_absolute_error=slope_error,derivative_frobenius=norm(local),derivative_bound=float(lip),
      witness=dict(quantum_branch_probability=signed,exact_classical_quadrature=[[str(j) for j in i] for i in classical],
                   same_terminal_mixture=True,terminal_effect_rank_before=1,terminal_effect_rank_after=2,
                   energy_population_ablation=energy_d,opposite_signed_response_gap=wrong_sign),
      effective_dimensions=dict(qubits=3,effect=40,signed_value=64,artificial_words=7),
      scope='Artificial CPU; no data-linear trainable encoder or native attainability check. Not science admission or trained classical comparison.',
      task_rows=0,task_forwards=0,optimizer_updates=0,gpu_allocations=0,
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(test(),indent=2,allow_nan=False))
