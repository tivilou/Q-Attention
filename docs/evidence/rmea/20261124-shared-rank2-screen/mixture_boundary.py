"""Small exact-formula example: relation-dependent mass gates can reverse margins.

Arbitrary unit real kets within C8, not a native circuit-reachability claim or a
trained-data experiment. No quantum advantage or novelty is claimed.
"""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parent
eps=1e-6
key=np.eye(8)[:2]
q=np.zeros((2,2,8))
for relation,masses in enumerate(((.01,.99),(.99,.01))):
    for component,mass in enumerate(masses):
        q[relation,component,component]=np.sqrt(mass)
        q[relation,component,2]=np.sqrt(1-mass)
assert np.allclose((q*q).sum(-1),1)
utility=np.array([[2.,-1.],[1.,-2.]])
probability=np.einsum('rkd,td->rkt',q,key)**2
raw=probability+eps;mass=raw.sum(-1)
component_logits=(raw*utility[:,None,:]).sum(-1)/mass
mixed_logits=(raw.mean(1)*utility).sum(-1)/raw.mean(1).sum(-1)
mass_formula=(component_logits*mass).sum(-1)/mass.sum(-1)
component_margins=component_logits[0]-component_logits[1]
mixed_margin=float(mixed_logits[0]-mixed_logits[1])
assert (component_margins>0).all() and mixed_margin<0
error=float(np.abs(mixed_logits-mass_formula).max());assert error<1e-12
record=dict(eps=eps,unit_query_norms=(q*q).sum(-1).tolist(),
    utilities=utility.tolist(),component_masses=mass.tolist(),
    component_logits=component_logits.tolist(),component_gold_margins=component_margins.tolist(),
    mixture_logits=mixed_logits.tolist(),mixture_gold_margin=mixed_margin,
    identity_error=error,
    interpretation='Both component models rank gold higher, while relation-dependent mass mixing reverses gold. Standard normalized-mixture phenomenon; not empirical evidence of the cause of any trained-model error, not native ansatz reachability, quantum necessity or new T.')
(r/'mixture-boundary.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record))
