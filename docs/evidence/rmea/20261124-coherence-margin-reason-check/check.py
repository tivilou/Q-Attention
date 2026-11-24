"""Necessary algebra screen of an unimplemented training idea, not a model test."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parent
def penalty(q,d):return max(0.,-(q-d))
# For gold-v-rival margins, jointly trainable D can worsen while Q stays wrong.
q=-1.;d=0.;before=penalty(q,d);d_bad=-2.;after=penalty(q,d_bad)
assert before==1 and after==0 and q<0
# Native Born+EPS readout witness: |phi>=|psi2>=(|0>+|j>)/sqrt2,
# |psi1>=|0>. Quantum probabilities (.5,1), diagonal (.5,.5).
# The native last RY followed by the fixed CNOT permutation realizes these
# states (j is that permutation's image of the rotated basis bit).
epsilon=1e-6
aq=(np.array([.5,1.])+epsilon)/(1.5+2*epsilon)
ad=np.array([.5,.5])
matrix=np.stack([aq,ad])
u_before=np.linalg.solve(matrix,np.array([-1.,0.]))
u_after=np.linalg.solve(matrix,np.array([-1.,-2.]))
measured_before=matrix@u_before;measured_after=matrix@u_after
assert np.allclose(measured_before,[-1,0]) and np.allclose(measured_after,[-1,-2])
# Stop-gradient fixes this local reference-only cheating direction, not the
# whole optimization trajectory: shared parameters still update D indirectly.
record=dict(question='Can nonnegative Q-minus-D margin gain be satisfied without repairing a wrong Q decision?',
 joint_reference_counterexample=dict(quantum_margin_before=q,quantum_margin_after=q,
 reference_margin_before=d,reference_margin_after=d_bad,penalty_before=before,penalty_after=after,
 quantum_still_wrong=True),
 native_readout_witness=dict(epsilon=epsilon,quantum_alpha=aq.tolist(),diagonal_alpha=ad.tolist(),
 utilities_before=u_before.tolist(),utilities_after=u_after.tolist(),
 quantum_and_reference_before=measured_before.tolist(),quantum_and_reference_after=measured_after.tolist(),
 max_residual=float(max(abs(measured_before-np.array([-1,0])).max(),abs(measured_after-np.array([-1,-2])).max())),
 scope='Two valid tokens, fixed reachable pure states and signed utilities, rival logit0; an affine utility map can realize both pairs on two distinct features. Not a measured descent direction of a trained checkpoint or general optimization theorem.'),
 conclusion='Naive joint-reference nonnegative-gain constraint is insufficient; do not implement/launch this objective as a valid repair.',
 stop_gradient_limit='Stops only direct reference gradient at that step; shared parameters still move D over the trajectory. Not a global no-cheating theorem.',
 fixed_reference_limit='Same-input fixed reference adds teacher provenance/storage and may forbid useful departures; does not establish quantum advantage or novelty.',
 status='NAIVE_OBJECTIVE_REJECTED_BEFORE_MODEL_CODE',new_model_updates=0,new_gpu=0,new_validation=0,new_test=0,new_cos=0)
(r/'margin-degeneracy.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
