"""Reconstruct PBR Eq1 and separate a product rule from a real bilinear rule."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parent
q=np.sqrt(.5)
measurement=np.array([[0,q,q,0],[.5,-.5,.5,.5],[.5,.5,-.5,.5],[q,0,0,-q]])
zero=np.array([1.,0.]);plus=np.array([q,q])
states=np.stack([np.kron(a,b) for a in (zero,plus) for b in (zero,plus)])
prob=abs(measurement@states.T)**2
orth=float(np.max(abs(measurement@measurement.T-np.eye(4))))
excluded=float(np.max(abs(np.diag(prob))))
assert orth<1e-12 and excluded<1e-12 and np.max(abs(prob.sum(0)-1))<1e-12
# A product local effect would give a rank-one table across the two choices.
determinant=float(np.linalg.det(prob[0].reshape(2,2)));assert abs(determinant)>1e-3
schmidt=[np.linalg.svd(x.reshape(2,2),compute_uv=False).tolist() for x in measurement]
assert all(s[-1]>.1 for s in schmidt)
rng=np.random.default_rng(20261125);max_error=0.
for complex_inputs in (False,True):
 for trial in range(128):
    a=rng.normal(size=2);b=rng.normal(size=2)
    if complex_inputs:a=a+1j*rng.normal(size=2);b=b+1j*rng.normal(size=2)
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    dense=abs(measurement@np.kron(a,b))**2
    factored=[]
    for row in measurement:
        m=row.reshape(2,2)
        real=a.real@m@b.real-a.imag@m@b.imag
        imag=a.real@m@b.imag+a.imag@m@b.real
        factored.append(real**2+imag**2)
    max_error=max(max_error,float(np.max(abs(dense-factored))))
assert max_error<1e-12
record=dict(status='PASS',source='arxiv:1111.3328v3 p3 Eq1, pp2-3 preparation order00,0+,+0,++',
    measurement_rows=measurement.tolist(),probability_outcomes_by_inputs=prob.tolist(),
    orthogonality_error=orth,excluded_diagonal_error=excluded,
    first_outcome_product_table_determinant=determinant,measurement_schmidt_values=schmidt,
    real_bilinear_representation_max_error=max_error,random_fixtures=256,
    new_model_fits=0,claim='Known nonproduct measurement, exactly evaluable as squared bilinear features. Neither simulation nor this known construction establishes or rejects task utility, hardware advantage, or a new T contribution.')
(r/'pbr-comparison-check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
