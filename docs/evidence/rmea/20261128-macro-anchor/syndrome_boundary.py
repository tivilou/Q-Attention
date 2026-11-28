"""Three-wire artificial example of syndrome protection and signal loss."""
import json
import numpy as np


def product(*parts):
    v=parts[0]
    for part in parts[1:]:
        v=np.kron(v,part)
    return v


def pure(v):
    return v[:,None]*v.conj()[None,:]


def run():
    i2=np.eye(2)
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    ident=np.eye(8)
    x1=product(x,i2,i2)
    z1=product(z,i2,i2)
    z2=product(i2,z,i2)
    p=(ident+z1@z2)/2
    rest=ident-p
    a,b=p,x1@rest
    assert np.array_equal(a.T@a+b.T@b,ident)
    assert np.array_equal(p@x1@p,np.zeros((8,8)))
    assert np.array_equal(p@z2@p,p@z2)
    assert np.array_equal(x1@z2,z2@x1)

    def recover(rho):
        return a@rho@a.T+b@rho@b.T

    def channel(e,rho,probability):
        return (1.-probability)*rho+probability*(e@rho@e.T)

    def effect(theta):
        state=np.zeros(8,dtype=complex)
        state[[0,6]]=[1.,np.exp(1j*theta)]
        return pure(state/np.sqrt(2))

    def evaluate(e,rho):
        return float(np.trace(e@rho).real)

    prepare=effect(0.)
    e=effect(0.)
    protected=[]; exposed=[]
    removed=[]
    for theta in [0.,np.pi/5,np.pi/2,3*np.pi/4,np.pi]:
        clean=effect(theta)
        for probability in (0.,.2,.7,1.):
            bad=channel(x1,clean,probability)
            protected.append(abs(evaluate(e,recover(bad))-evaluate(e,clean)))
            exposed.append(abs(evaluate(e,bad)-(1.-probability)*evaluate(e,clean)))
        unitary=np.cos(theta/2)*ident-1j*np.sin(theta/2)*x1
        recovered=recover(unitary@prepare@unitary.conj().T)
        removed.append(float(np.max(np.abs(recovered-prepare))))
    same_sector=channel(z1,prepare,.25)
    assert abs(evaluate(e,recover(same_sector))-.75)<2e-15
    for xerr in (protected,exposed,removed):
        assert max(xerr)<2e-15

    rng=np.random.default_rng(20261128)
    effects=np.stack([effect(v) for v in np.linspace(.1,5.1,40)])
    dual=effects+x1[None,:,:]@effects@x1[None,:,:]
    scores=[]
    scores_dual=[]
    for _ in range(64):
        mat=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
        rho=mat@mat.conj().T
        rho/=np.trace(rho).real
        scores.append([evaluate(effect,recover(rho)) for effect in effects])
        scores_dual.append([evaluate(effect,rho) for effect in dual])
    errors=float(np.max(np.abs(np.array(scores)-np.array(scores_dual))))
    assert errors<2e-15
    return dict(status='PASS_SYNTHETIC_CORRECTION_AND_EXACT_PULLBACK_BOUNDARY',
                known_correctable_noise='random Pauli X on the first wire',
                known_commuting_signal='phase difference on logical basis |000> / |110>',
                max_preserved_signal_error=max(protected),max_uncorrected_formula_error=max(exposed),
                max_same_syndrome_coherent_signal_erasure_error=max(removed),
                same_parity_phase_flip_recovered_probability=evaluate(e,recover(same_sector)),
                rank_of_each_pulled_back_effect=[int(np.linalg.matrix_rank(q)) for q in dual],
                pulled_back_effect_max_error=errors,psd_fixtures=64,independent_binary_effects=40,
                claim_scope='Known discrete toy channel only. No training, text nuisance evidence, quantum necessity, new theory or deployment.',
                task_forwards=0,optimizer_updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))