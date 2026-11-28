"""A three-mode, two-photon resource within the original six complex encoding coordinates."""
import itertools
import json
import math

import numpy as np


def kernels(delta):
    intensity=np.abs(delta)**2
    active=intensity[..., :3]
    envelope=np.exp(-intensity.sum(-1))
    laguerre=1-2*active+active**2/2
    correction=np.abs((delta[..., :3]**2).sum(-1))**2/6
    coherent=envelope*(1-2*active.mean(-1)+correction)**2
    pair_cross=sum((active[...,i]*active[...,j])**2/2 for i,j in itertools.combinations(range(3),2))
    diagonal=envelope*((laguerre**2).sum(-1)+pair_cross)/3
    matched=np.exp(-active.sum(-1)*7/3-intensity[...,3:].sum(-1))
    return np.stack((coherent,diagonal,matched),axis=-1)


def displacement(delta, cutoff):
    lower=np.diag(np.sqrt(np.arange(1,cutoff)),1)
    h=-1j*(delta*lower.T-delta.conjugate()*lower)
    eig,u=np.linalg.eigh(h)
    return (u*np.exp(1j*eig))@u.conj().T


def independent(delta, cutoff):
    local=[displacement(z,cutoff) for z in delta]
    states=[tuple(2*(i==j) for i in range(6)) for j in range(3)]
    compression=np.array([[np.prod([m[i,j] for m,i,j in zip(local,a,b)]) for b in states] for a in states])
    q=np.abs(compression.sum()/3)**2
    d=np.vdot(compression,compression).real/3
    return np.array([q,d])


def scalar_replay(delta):
    # Independent real quartic reference: neither circuit nor quantum output.
    x,y=delta[..., :3].real,delta[..., :3].imag
    u=(x*x-y*y).sum(-1)
    v=2*(x*y).sum(-1)
    s=(x*x+y*y).sum(-1)
    whole=(np.square(delta.real)+np.square(delta.imag)).sum(-1)
    return np.exp(-whole)*(1-2*s/3+(u*u+v*v)/6)**2


def run():
    rng=np.random.default_rng(20261128)
    selected=(rng.normal(size=(64,6))+1j*rng.normal(size=(64,6)))*.33
    exact=kernels(selected)
    low=np.stack([independent(d,20) for d in selected])
    high=np.stack([independent(d,24) for d in selected])
    assert np.max(np.abs(low-high))<1e-12 and np.max(np.abs(high-exact[:,:2]))<1e-12
    assert np.all(exact>=0) and np.all(exact<=1+1e-12)
    a=np.array([1,1,1,0,0,0],dtype=complex)/np.sqrt(2)
    b=a*np.exp(1j*np.arange(6)*np.pi/3)
    first=kernels(a)
    second=kernels(b)
    assert abs(first[0]-9*math.exp(-1.5)/64)<1e-14
    assert second[0]<1e-30 and np.max(np.abs(first[1:]-second[1:]))<1e-15
    assert np.max(np.abs(kernels(np.zeros(6))-1))<1e-12
    for i in range(6):
        basis=np.eye(6)[i]
        h=1e-4
        slope=(1-kernels(h*basis))/h**2
        assert np.max(np.abs(slope-(7/3 if i<3 else 1)))<1e-6
    # Pure two-boson coefficient rank is invariant under passive unitary
    # congruence. A product occupation of exactly two photons has rank one
    # (2 on the same mode) or two (1 in two modes).
    s=np.diag([1,1,1,0,0,0])/np.sqrt(3)
    u,_=np.linalg.qr(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))
    rank=int(np.linalg.matrix_rank(u@s@u.T,tol=1e-12))
    assert rank==3

    keys=(rng.normal(size=(9,6))+1j*rng.normal(size=(9,6)))*.24
    queries=(rng.normal(size=(40,6))+1j*rng.normal(size=(40,6)))*.24
    delta=keys[None]-queries[:,None]
    values=rng.normal(size=(9,64))
    read=rng.normal(size=(40,64))
    exact_prob=kernels(delta)[...,0]
    classical_prob=scalar_replay(delta)
    def evaluate(weights):
        w=weights+1e-6
        v=(w/w.sum(1)[:,None])@values
        return np.sum(v*read,axis=-1),v
    scores,pools=evaluate(exact_prob)
    exact_real,real_pools=evaluate(classical_prob)
    dscore,_=evaluate(kernels(delta)[...,1])
    assert np.max(np.abs(scores-exact_real))<2e-14
    assert np.max(np.abs(scores-dscore))>1e-4
    return dict(status='PASS_FIXED_TWO_PHOTON_CROSS_MODE_WITNESS',all_40_classes=True,value_dim=64,
                total_photons=2,active_coherent_modes=3,passive_two_boson_rank=rank,
                source_kernels_at_equal_intensity=[first.tolist(),second.tolist()],
                gaussian_curvature_match_and_origin_checked=True,
                projected_displacement_fixture_count=len(selected),unitary_cutoffs=[20,24],
                max_operator_error=float(np.max(np.abs(high-exact[:,:2]))),
                max_two_cutoff_disagreement=float(np.max(np.abs(low-high))),
                max_exact_real_replay_logit_delta=float(np.max(np.abs(scores-exact_real))),
                max_exact_real_replay_pool_delta=float(np.max(np.abs(pools-real_pools))),
                artificial_dephased_logit_delta=float(np.max(np.abs(scores-dscore))),
                inference_mechanism='Known displacement-overlap framework; fixed-polynomial classical arithmetic exactly matches scores. No hardware or task advantage.',
                new_task_forwards=0,optimizer_updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))