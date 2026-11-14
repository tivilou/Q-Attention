"""Independent dense circuit and exact extremal fixtures."""
import math
import numpy as np
import torch
from legacy.rmea import circuit
from geometry import dense_states,describe

def run_checks():
    rng=np.random.default_rng(5603);a=rng.normal(0,.4,(5,2,3,2))
    v=dense_states(a);w=circuit(torch.from_numpy(a)).detach().numpy()
    np.testing.assert_allclose(v,w,atol=2e-14,rtol=0)
    tight=np.repeat(np.eye(8,dtype=np.complex128),5,axis=0)
    t=describe(tight);np.testing.assert_allclose(t['eigenvalues'],np.ones(8)/8,atol=1e-14)
    assert abs(t['inverse_purity_effective_rank']-8)<1e-12 and abs(t['normalized_frame_excess'])<1e-12
    # Tight frames need not separate all pairs: this optimal frame has duplicates.
    assert t['pairs_overlap_gt_point99']==80
    same=np.tile(np.eye(8,dtype=np.complex128)[0],(40,1));s=describe(same)
    assert s['inverse_purity_effective_rank']==1 and s['normalized_frame_excess']==7 and s['pairs_overlap_gt_point99']==780
    x=rng.normal(size=(40,8))+1j*rng.normal(size=(40,8));x/=np.linalg.norm(x,axis=1,keepdims=True)
    q,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    y=(x*np.exp(1j*rng.normal(size=(40,1))))[rng.permutation(40)]@q
    dx,dy=describe(x),describe(y)
    for k in ('eigenvalues','purity','normalized_frame_excess','offdiag_mean_overlap_squared','offdiag_quantiles'):
        np.testing.assert_allclose(dx[k],dy[k],atol=2e-12,rtol=0)
    assert dx['offdiag_mean_overlap_squared']>=dx['welch_lower_bound']-1e-12
    return dict(status='PASS',checks=['independent dense NumPy/legacy Torch circuit parity','normalized states',
        'tight repeated-basis analytic optimum','tight optimum can retain duplicate pairs','identical-state analytic maximum',
        'row phase/permutation/global unitary invariance','purity-trace-Gram identity','Welch lower bound'],
        max_circuit_error=float(np.abs(v-w).max()))
