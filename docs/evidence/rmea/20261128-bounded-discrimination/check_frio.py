"""Artificial 40-state FRIO optimum vs all post-processing of fixed outcomes."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
import numpy as np

D,N=8,40
P=(F(3,10),)+(F(1,10),)*7
ETA=(F(9,10),F(4,5),F(3,5))

def ensure(a,b,tol=2e-12):
    e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert e<tol,e;return e

def waterfill(p,eta):
    assert 0<eta<=sum(p)
    active=list(p); free=len(p); total=eta
    for b in sorted(p):
        level=total/free
        if level<=b:return [min(q,level) for q in p]
        total-=b;free-=1
    raise ValueError('invalid capacity')

def dense(s,m):return np.array([[np.vdot(v,q@v).real for q in m] for v in s])

def test():
    # W columns d basis levels; W has N equally spaced phase rows.
    w=np.exp(2j*np.pi*np.arange(N)[:,None]*np.arange(D)[None,:]/N)
    p=np.array(P,float); states=w*np.sqrt(p)[None,:];seed=w/np.sqrt(N)
    def effect(v):return np.outer(v,v.conj())
    fixed=np.array([effect(v) for v in seed]);errors={}
    errors['unfiltered_povm']=ensure(fixed.sum(axis=0),np.eye(D))
    base=dense(states,fixed)
    optimum=(np.sqrt(p).sum()**2/N)
    errors['fixed_posterior']=ensure(base.max(axis=0),optimum)
    errors['constant_outcome_prior']=ensure(base.mean(axis=0),1/N)
    assert (base.argmax(axis=0)==np.arange(N)).all()
    # Any post-processing stochastic kernel is bounded because each classical
    # outcome has the same best conditional correctness. Include remapping.
    rng=np.random.default_rng(20261128);K=rng.uniform(size=(N,N+1));K/=K.sum(axis=1,keepdims=True)
    post=base@K;mass=post[:,:N].sum()/N
    assert np.trace(post[:,:N])/N<=mass*optimum+1e-14

    out=[]
    for eta in ETA:
        x=waterfill(P,eta);assert sum(x)==eta and all(0<a<=b for a,b in zip(x,P))
        level=max(x);assert all(a==min(b,level) for a,b in zip(x,P))
        filt=np.sqrt(np.array([j/z for j,z in zip(x,P)],float));A=np.diag(filt**2)
        M=np.array([effect(v*filt) for v in seed]);p_eff=dense(states,M)
        # Second evaluation is pure-state interferometer Fourier amplitudes, no POVM matrix.
        fourier=abs(((states*filt)@seed.conj().T))**2
        e=ensure(M.sum(axis=0),A); e=max(e,ensure(p_eff,fourier),ensure(p_eff.sum(axis=1),float(eta)))
        R=np.eye(D)-A;e=max(e,ensure(dense(states,R[None])[:,0],1-float(eta)))
        val=sum(math.sqrt(z) for z in x)**2/(N*float(eta));e=max(e,ensure(np.diag(p_eff),float(eta)*val));
        assert val>optimum and val<=D/N+1e-14
        assert np.linalg.eigvalsh(R).min()>=-1e-14
        # Full phase-kill gives the same population for each class for any effect;
        # here explicit computed conclusive probabilities are constant eta/N.
        pops=abs(states)**2
        e=max(e,ensure(pops@M.diagonal(axis1=1,axis2=2).real.T,float(eta)/N))
        # With original 64-dimensional signed values, use all conclusive masses;
        # do not map the reject event to semantic no_relation.
        values=rng.normal(size=(N,64));valid=np.ones(N,bool);valid[[2,21]]=False; eps=1e-6
        v0=(np.where(valid[:,None],p_eff+eps,0).T@values)/np.where(valid[:,None],p_eff+eps,0).sum(axis=0)[:,None]
        v1=(np.where(valid[:,None],fourier+eps,0).T@values)/np.where(valid[:,None],fourier+eps,0).sum(axis=0)[:,None]
        e=max(e,ensure(v0,v1)); assert np.linalg.norm(v0)>0
        out.append(dict(coverage=str(eta),waterfilled_mass=[str(z) for z in x],conditioning_correct=val,
                       optimized_wrong=float(1-val),exact_fixed_record_best_correct=optimum,
                       max_matrix_fourier_pool_error=e,phase_kill_correct=1/N))
    assert out[1]['waterfilled_mass']==['1/10']*8 and abs(out[1]['conditioning_correct']-.2)<1e-14
    assert abs(out[2]['conditioning_correct']-.2)<1e-14

    # Isotropic filtering has an exactly accounted-for effect from the EPS floor.
    q=base[:,0];eps=1e-6;beta=.8;n=len(q)
    actual=(beta*q+eps)/(beta*q.sum()+n*eps);mix=beta*q.sum()/(beta*q.sum()+n*eps)
    errors['eps_isotropic_mix']=ensure(actual,mix*q/q.sum()+(1-mix)/n)
    # No-reject kernel may be classical but this exact postprocessing bound only
    # assumes the unfiltered class outcome as its *sole* input.
    return dict(status='PASS_CYCLIC_FRIO_WATERFILL_AND_RECORD_BOUND',classes=N,dimension=D,fixture_prior='uniform class prior; fixed cyclic populations',
       coverage_cases=out,errors=errors,uniform_conclusive_class_accuracy_cap=D/N,
       fixed_outcome_posterior=optimum,scope='Artificial cyclic source only; exact replay sees amplitudes, not a trained independent comparator or task utility. Physical inconclusive is separate from all semantic classes.',
       task_rows=0,task_forwards=0,optimizer_updates=0,gpu_allocations=0,
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(test(),indent=2,allow_nan=False))
