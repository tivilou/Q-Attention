"""Synthetic transfer checks, not new theory, training or empirical utility.

1. Matrix sign cannot be interchanged with averaging pure-state pairs.
2. A uniform sentence density omits the native classifier's signed-value pairing.
3. Native logits retain an exact signed-operator fractional representation.
"""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import torch
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r.parent/'qattn-relation-frame-20261125/source'))
from baselines.retacred.rmea import RMEAHead,EPS


def projector(v):
    return np.outer(v,v.conj())


def matrix_sign(a):
    w,v=np.linalg.eigh(a)
    assert np.min(np.abs(w))>1e-8, 'Counterexample avoids sign-at-zero conventions'
    return (v*np.sign(w)) @ v.conj().T


zero=np.array([1.,0.]);one=np.array([0.,1.]);plus=(zero+one)/math.sqrt(2)
pa=projector(zero);pb=[projector(one),projector(plus)]
sigma=sum(pb)/2;delta=pa-sigma
true_sign=matrix_sign(delta)
averaged_pair_sign=sum(matrix_sign(pa-b) for b in pb)/2
qz=3/math.sqrt(73)
query=np.array([math.sqrt((1+qz)/2),math.sqrt((1-qz)/2)])
true_output=float(query @ true_sign @ query)
proxy_output=float(query @ averaged_pair_sign @ query)
expected_true=1/math.sqrt(730)
expected_proxy=(3-5/math.sqrt(2))/(2*math.sqrt(73))
assert abs(true_output-expected_true)<1e-14
assert abs(proxy_output-expected_proxy)<1e-14
assert true_output>0>proxy_output
assert np.allclose(np.linalg.norm(query),1)
fixed_sample_two_copies=sum(np.kron(b,b) for b in pb)/2
independent_mixture_two_copies=np.kron(sigma,sigma)
copy_error=float(np.linalg.norm(fixed_sample_two_copies-independent_mixture_two_copies))
assert copy_error>.1
high_rank=np.diag([.5,.5,-.5,-.5])
assert np.linalg.matrix_rank(high_rank)==4

# Fidelity risk identity holds for its own linear-output objective, up to the
# chosen equal-class risk normalization; no CE/Macro/generalization implication.
fidelity_difference=float(np.trace(delta @ delta))
kernel_difference=float(np.trace(pa@pa)+np.trace(sigma@sigma)-2*np.trace(pa@sigma))
assert abs(fidelity_difference-kernel_difference)<1e-14

# Exact native counterexample: only last-layer wire2 RY is nonzero. Encoding
# reads feature0, value reads feature1; swapping value pairing leaves keys intact.
torch.set_num_threads(1);torch.manual_seed(13)
model=RMEAHead('quantum').double()
with torch.no_grad():
    for p in model.parameters():p.zero_()
    gate_index=10
    model.prepare.weight[gate_index,0]=math.atanh(.5)
    model.effect[0,1,2,0]=math.atanh(.5)
    model.value.weight[0,1]=1.
    model.decoder[0,0]=1.
h=torch.zeros(2,2,1024,dtype=torch.double)
h[:,:,0]=torch.tensor([[1.,-1.],[1.,-1.]],dtype=torch.double)
h[:,:,1]=torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.double)
roles=torch.zeros(2,2,4,dtype=torch.double);mask=torch.ones(2,2,dtype=torch.bool)
with torch.no_grad():
    details=model(h,roles,mask,details=True)
    psi=details['psi'];rho=psi[..., :,None]*psi.conj()[...,None,:]
    sentence_density=rho.mean(1)
    density_error=float((sentence_density[0]-sentence_density[1]).abs().max())
    assert density_error==0
    outputs=details['logits'][:,0].tolist()
    assert abs(outputs[0]-1/(1+2*EPS))<1e-14
    assert abs(outputs[1]+1/(1+2*EPS))<1e-14

# Independently assemble signed operators, INCLUDING EPS exactly as in native.
    utilities=details['utilities']
    identity=torch.eye(8,dtype=torch.complex128)
    atoms=rho+EPS*identity
    C=atoms.sum(1)
    A=torch.einsum('brt,btij->brij',utilities.to(torch.complex128),atoms)
    phi=details['phi']
    E=phi[:,:,None]*phi.conj()[:,None,:]
    denominator=torch.einsum('rij,bji->br',E,C).real
    numerator=torch.einsum('rij,brji->br',E,A).real
    reconstructed=numerator/denominator+model.bias
    native_error=float((reconstructed-details['logits']).abs().max())
    assert native_error<1e-14

# Random all40 complex states, signed utilities, masks, EPS: direct token sum
# versus fractional operators and the pairwise cross-multiplied margin.
rng=np.random.default_rng(13)
max_fractional_error=0.;max_margin_error=0.
for trial in range(64):
    n=1+trial%9
    psi=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8))
    psi/=np.linalg.norm(psi,axis=1)[:,None]
    phi=rng.normal(size=(40,8))+1j*rng.normal(size=(40,8))
    phi/=np.linalg.norm(phi,axis=1)[:,None]
    u=rng.normal(size=(40,n));bias=rng.normal(size=40)
    probabilities=np.abs(phi.conj()@psi.T)**2
    raw=probabilities+EPS
    z=(raw*u).sum(-1)/raw.sum(-1)+bias
    atoms=np.einsum('ti,tj->tij',psi,psi.conj())+EPS*np.eye(8)
    C=atoms.sum(0);A=np.einsum('rt,tij->rij',u,atoms)
    E=np.einsum('ri,rj->rij',phi,phi.conj())
    Z=np.einsum('rij,ji->r',E,C).real
    N=np.einsum('rij,rji->r',E,A).real
    max_fractional_error=max(max_fractional_error,float(np.max(np.abs(z-(N/Z+bias)))))
    y,c=trial%40,(trial*7+1)%40
    lifted=np.kron(A[y],C)-np.kron(C,A[c])+(bias[y]-bias[c])*np.kron(C,C)
    expectation=np.trace(np.kron(E[y],E[c]) @ lifted).real
    margin=expectation/(Z[y]*Z[c])
    max_margin_error=max(max_margin_error,abs(float(margin-(z[y]-z[c]))))
assert max_fractional_error<1e-13 and max_margin_error<1e-13

result=dict(status='PASS',scope='Synthetic mathematical transfer-boundary audit only',
    pairwise_sign_counterexample=dict(ensemble_sign_output=true_output,pure_pair_average_output=proxy_output,
        analytic_outputs=[expected_true,expected_proxy],decisions_opposite=True,
        matrix_error=float(np.linalg.norm(true_sign-averaged_pair_sign)),
        fixed_sample_copy_vs_independent_copy_error=copy_error,
        mixed_ensemble_rank_example=4,pure_pair_rank_bound=2,
        limit='Refutes generic interchange/sign proxy. Does not deny correctly prepared repeated independent ensemble copies or Helstrom theorem.'),
    known_fidelity_HS_identity_error=abs(fidelity_difference-kernel_difference),
    native_value_pairing_counterexample=dict(sentence_density_error=density_error,native_relation_logits=outputs,
        exact_expected_logits=[1/(1+2*EPS),-1/(1+2*EPS)],fractional_operator_error=native_error,
        limit='An unweighted sentence density alone is insufficient for this native parameter setting; not a universal no-go for all embeddings.'),
    random_trials=64,all40_fractional_error=max_fractional_error,pairwise_lifted_margin_error=max_margin_error,
    newT=False,task_rows=0,optimizer_updates=0,gpu_allocations=0,
    source_identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        r.parent/'qattn-relation-frame-20261125/source/baselines/retacred/rmea.py',Path(__file__)]})
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
