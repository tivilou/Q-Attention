"""Known optimizer limitation, independent of experiment outcomes; no fit/data."""
from pathlib import Path
import json
import numpy as np

u=np.array([1.,-2.])/np.sqrt(5);n=np.array([2.,1.])/np.sqrt(5)
f=np.outer(u,u);gradient=u
# First-Adam-like elementwise normalization can leave the observable tangent.
delta=np.sign(gradient);damping=.01*np.trace(f)/2+1e-8
w,v=np.linalg.eigh(f)
shaped=v@((np.maximum(w,0)+damping)**(-.5)*(v.T@delta))
shaped*=np.linalg.norm(delta)/np.linalg.norm(shaped)
def null_fraction(x):return float((n@x)**2/(x@x))
record=dict(original_null_step_fraction=null_fraction(delta),
    fisher_shaped_null_step_fraction=null_fraction(shaped),
    gradient_null_component=float(abs(n@gradient)),step_norm_error=float(abs(np.linalg.norm(delta)-np.linalg.norm(shaped))),
    effect='Norm-matched damped inverse may amplify an unobservable component introduced by elementwise Adam scaling.',
    scope='Known PSD/optimizer geometry example, not a new theorem or evidence that any actual arm behaves this way.',
    policy='Keep the fixed experiment and all outcomes. Even an 8-gate screen pass cannot justify a contribution if unchanged classical controls are missing from the formal study. No post-result damping/power/range-projection rescue.')
assert record['gradient_null_component']<1e-15 and record['step_norm_error']<1e-15
assert record['original_null_step_fraction']<.11 and record['fisher_shaped_null_step_fraction']>.95
(Path(__file__).parent/'null-direction.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
