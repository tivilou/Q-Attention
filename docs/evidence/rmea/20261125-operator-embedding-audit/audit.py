"""Finite native-POVM map, tomography boundary, and a printed-bound counterexample."""
from pathlib import Path
import argparse, ast, hashlib, json, math, itertools
from fractions import Fraction as F
import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--native-source',required=True)
parser.add_argument('--out',required=True)
a=parser.parse_args()
source=Path(a.native_source).read_bytes();sha=hashlib.sha256(source).hexdigest()
assert sha=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='circuit')
ns={'torch':torch,'QUBITS':3,'DEPTH':2}
exec(compile(ast.Module(body=[node],type_ignores=[]),a.native_source,'exec'),ns)
rng=np.random.default_rng(260525146)
effect_angles=rng.uniform(-2.4,2.4,(40,2,3,2))
key_angles=rng.uniform(-2.4,2.4,(10,2,3,2))
effects=ns['circuit'](torch.tensor(effect_angles,dtype=torch.float64)).numpy()
keys=ns['circuit'](torch.tensor(key_angles,dtype=torch.float64)).numpy()
eps=1e-6;eye=np.eye(8)
B=np.einsum('ti,tj->tij',keys,keys.conj())+eps*eye
C=B.sum(0);e,V=np.linalg.eigh(C)
sqrt=(V*np.sqrt(e))@V.conj().T;invsqrt=(V/np.sqrt(e))@V.conj().T
povm=np.einsum('ij,tjk,kl->til',invsqrt,B,invsqrt)
chi=(sqrt@effects.T).T
chi/=np.sqrt(np.einsum('ri,ri->r',chi.conj(),chi).real)[:,None]
alpha=abs(effects.conj()@keys.T)**2+eps;alpha/=alpha.sum(-1,keepdims=True)
conditional=np.einsum('ri,tij,rj->rt',chi.conj(),povm,chi).real
u=np.sin(np.arange(1,41)[:,None]*np.arange(1,11)[None,:])
moment=np.einsum('rt,tij->rij',u,povm)
z=np.einsum('ri,rij,rj->r',chi.conj(),moment,chi).real
map_errors=dict(povm_sum=float(np.max(abs(povm.sum(0)-eye))),
    minimum_povm_eigenvalue=float(np.linalg.eigvalsh(povm).min()),
    conditional_alpha=float(np.max(abs(conditional-alpha))),
    signed_readout=float(np.max(abs(z-(alpha*u).sum(-1)))))
assert max(map_errors[k] for k in ['povm_sum','conditional_alpha','signed_readout'])<1e-11
assert map_errors['minimum_povm_eigenvalue']>0

# Orthonormal real basis of traceless Hermitian 8x8 matrices.
basis=[]
for k in range(1,8):
    v=np.zeros(8);v[:k]=1;v[k]=-k;basis.append(np.diag(v)/math.sqrt(k*(k+1)))
for i in range(8):
    for j in range(i+1,8):
        R=np.zeros((8,8),complex);R[i,j]=R[j,i]=1/math.sqrt(2);basis.append(R)
        J=np.zeros((8,8),complex);J[i,j]=1j/math.sqrt(2);J[j,i]=-1j/math.sqrt(2);basis.append(J)
basis=np.array(basis);assert len(basis)==63
E=np.einsum('ri,rj->rij',effects,effects.conj())
D=np.einsum('rij,bji->rb',E,basis).real
_,s,vh=np.linalg.svd(D,full_matrices=True)
rank=int(np.sum(s>1e-11));assert rank==40
H=np.einsum('b,bij->ij',vh[-1],basis)
H/=np.max(abs(np.linalg.eigvalsh(H)))
rho_plus=eye/8+H/16;rho_minus=eye/8-H/16
means_plus=np.einsum('ij,rji->r',rho_plus,E).real
means_minus=np.einsum('ij,rji->r',rho_minus,E).real
collision=dict(traceless_dimension=63,measurement_rows=40,numeric_rank=rank,
    nullity=63-rank,smallest_nonzero_singular_value=float(s[-1]),
    min_eigenvalue_plus=float(np.linalg.eigvalsh(rho_plus).min()),
    min_eigenvalue_minus=float(np.linalg.eigvalsh(rho_minus).min()),
    trace_plus=float(np.trace(rho_plus).real),trace_minus=float(np.trace(rho_minus).real),
    density_frobenius_separation=float(np.linalg.norm(rho_plus-rho_minus)),
    all40_measurement_difference=float(np.max(abs(means_plus-means_minus))))
assert collision['all40_measurement_difference']<1e-13
assert min(collision['min_eigenvalue_plus'],collision['min_eigenvalue_minus'])>.06

# Exact 8-outcome enumeration of a complete q=2 Pauli design.
p_minus=F(1,1000);az=F(499,500);scale=F(10);threshold=F(10)
cases=[];tail=F(0);mean=F(0);variance=F(0)
for x,y,z0 in itertools.product([-1,1],repeat=3):
    prob=F(1,4)*(p_minus if z0==-1 else 1-p_minus)
    contrast=-scale*(z0-az)
    cases.append(dict(outcomes=[x,y,z0],probability=str(prob),contrast=str(contrast)))
    tail+=prob*(contrast>threshold);mean+=prob*contrast;variance+=prob*contrast**2
assert tail==F(1,1000) and mean==0 and variance==F(999,2500)
h=lambda x:(1+x)*math.log1p(x)-x
v=float(variance);t=float(threshold);jump=max(float(F(row['contrast'])) for row in cases)
printed_log_bound=-v*h(t/v)
corrected_log_bound=-v/jump**2*h(jump*t/v)
assert math.log(float(tail))>printed_log_bound
assert float(tail)<=math.exp(corrected_log_bound)
bound=dict(paper='2605.25146v1',locator='Theorem4.25 p21; proof p45',
    dimension=2,observables=['PauliX','PauliY','PauliZ'],kernel='Dirac on +/-1',
    complete_design=True,shots_per_observable=1,state_diagonal=['999/1000','1/1000'],
    direction='-10 PauliZ',threshold='10',exact_tail=str(tail),exact_mean=str(mean),
    exact_variance=str(variance),enumeration=cases,largest_positive_increment=jump,
    printed_log_upper_bound=printed_log_bound,printed_upper_bound=math.exp(printed_log_bound),
    corrected_known_Bennett_log_bound=corrected_log_bound,
    corrected_known_Bennett_upper_bound=math.exp(corrected_log_bound),
    conclusion='Printed arbitrary-direction inequality has a counterexample; no claim about all source results.')
pack=lambda A:dict(real=A.real.tolist(),imag=A.imag.tolist())
out=dict(status='PASS_AUDIT_COUNTEREXAMPLES_NOT_SOURCE_THEOREM',native_source_sha256=sha,
    finite_token_povm=map_errors,full_mixed_state_tomography=collision,printed_bound=bound,
    synthetic_inputs=dict(effect_angles=effect_angles.tolist(),key_angles=key_angles.tolist(),
        null_operator=pack(H),rho_plus=pack(rho_plus),rho_minus=pack(rho_minus)),
    formula_is_new_method=False,new_T_claim=False,task_rows=0,trained_head_forwards=0,
    training_updates=0,gpu_allocations=0,
    limitations=['Port/mixed-state witnesses, not actual input/encoder-image or pure-state collision claims.',
        'Full tomography is not required for classification; nullity does not explain empirical errors.',
        'Known POVM congruence and Bennett range factor are not new quantum theory.'])
Path(a.out).write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ['status','finite_token_povm','full_mixed_state_tomography','printed_bound']}))
