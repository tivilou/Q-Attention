"""Native synthetic Pauli-shell, positive marginal and Mercer-transfer audit."""
from pathlib import Path
import json,hashlib,sys,types,importlib.util,itertools,math,time
import numpy as np
import torch
torch.set_num_threads(1)
r=Path(__file__).resolve().parent;p=json.loads((r/'numerical-protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
source=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred')
assert sha((source/'rmea.py').read_bytes())==p['native_source_sha256']
assert sha((source/'frozen_feature_head.py').read_bytes())=='f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9'
pkg=types.ModuleType('native_trace_port');pkg.__path__=[str(source)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.rmea',source/'rmea.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
start=time.monotonic();rng=np.random.default_rng(p['seed'])
strings=np.array(list(itertools.product(range(4),repeat=3)));weight=(strings!=0).sum(-1)
local=np.array([np.eye(2),[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
paulis=np.array([np.kron(np.kron(local[a],local[b]),local[c]) for a,b,c in strings])
ka=rng.uniform(-2.6,2.6,(p['random_keys'],2,3,2));qa=rng.uniform(-2.6,2.6,(p['random_queries'],2,3,2))
# Output |sign0,sign1,sign2>_X characters transform by A^-T under the fixed CNOT ring.
A=np.array([[0,1,1],[1,1,0],[1,1,1]])
characters=np.array(list(itertools.product(range(2),repeat=3)))
inputs=characters@A % 2
xa=np.zeros((8,2,3,2));xa[:,1,:,0]=np.where(inputs==0,1,-1)*math.pi/2
ka=np.concatenate([xa,ka])
ks=native.circuit(torch.tensor(ka,dtype=torch.float64)).detach().numpy()
qs=native.circuit(torch.tensor(qa,dtype=torch.float64)).detach().numpy()
rho=np.einsum('ni,nj->nij',ks,ks.conj());sigma=np.einsum('ni,nj->nij',qs,qs.conj())
kr=np.einsum('aij,nji->na',paulis,rho).real;qr=np.einsum('aij,nji->na',paulis,sigma).real
shell={str(H):kr[:,weight==H]@qr[:,weight==H].T/math.sqrt(math.comb(3,H)*3**H) for H in range(4)}
shellgram={str(H):kr[:,weight==H]@kr[:,weight==H].T/math.sqrt(math.comb(3,H)*3**H) for H in range(4)}
prob=abs(ks.conj()@qs.T)**2
recovered=sum(math.sqrt(math.comb(3,H)*3**H)*shell[str(H)] for H in range(4))/8
global_error=float(abs(prob-recovered).max());assert global_error<p['tolerance']
z=np.all(np.isin(strings,[0,3]),axis=-1)
d=abs(ks)**2@abs(qs.T)**2
dephased_error=float(abs(d-kr[:,z]@qr[:,z].T/8).max());assert dephased_error<p['tolerance']

def marginal(state,keep):
    rest=[j for j in range(3) if j not in keep]
    v=state.reshape(2,2,2).transpose(list(keep)+rest).reshape(2**len(keep),-1)
    return v@v.conj().T

reduced={};errors={}
for S in [1,2,3]:
    keeps=list(itertools.combinations(range(3),S));total=np.zeros((len(ks),len(qs)))
    for keep in keeps:
        a=np.array([marginal(v,keep) for v in ks]);b=np.array([marginal(v,keep) for v in qs])
        total+=np.einsum('nij,mji->nm',a,b).real/len(keeps)
    coefficients=np.array([math.comb(3-int(h),S-int(h)) if h<=S else 0 for h in weight])/(2**S*math.comb(3,S))
    expanded=(kr*coefficients)@qr.T
    errors[str(S)]=float(abs(total-expanded).max());assert errors[str(S)]<p['tolerance']
    assert total.min()>-p['tolerance'] and total.max()<1+p['tolerance']
    reduced[str(S)]=dict(mean_hs=total.tolist(),sourceS_scale=math.sqrt(math.comb(3,S)),
       sourceS_Pauli_weight_l2_squared=float(math.comb(3,S)*sum(coefficients**2)),
       min=float(total.min()),max=float(total.max()))

# Exact source Eq42 H=1 has +/-1 entries on the legal X-basis ports.
expected_basis=np.array([[np.prod([(-1)**(c[j]*x[j]) for j in range(3)])/math.sqrt(8) for x in itertools.product(range(2),repeat=3)] for c in characters])
basis_error=float(abs(ks[:8]-expected_basis).max());assert basis_error<p['tolerance']
h1=shellgram['1'][0,[0,7]];assert abs(h1[0]-1)<p['tolerance'] and abs(h1[1]+1)<p['tolerance']
raw=h1+native.EPS;alpha=raw/raw.sum();u=np.array([1.,-1.]);bad_logit=float(alpha@u)
original_raw=np.array([1.,0.])+native.EPS;original_alpha=original_raw/original_raw.sum()
assert raw[1]<0 and alpha[1]<0 and bad_logit>9e5

# Two native legal states form orthogonal rank1 Mercer eigenoperators with distinct eigenvalues.
pair=rho[[0,7]];mu=np.array([1/3,2/3])
omega=sum(w*np.outer(v.reshape(-1),v.reshape(-1).conj()) for w,v in zip(mu,pair))
eigenvalues,eigenvectors=np.linalg.eigh(omega)
residuals=[]
for w,v in zip(mu,pair):residuals.append(float(abs(omega@v.reshape(-1)-w*v.reshape(-1)).max()))
assert max(residuals)<p['tolerance']
assert np.allclose(eigenvalues[-2:],mu,atol=1e-12)
rank=[int(np.linalg.matrix_rank(v,tol=1e-10)) for v in pair]
normalized_pauli_ranks=[int(np.linalg.matrix_rank(v/math.sqrt(8),tol=1e-10)) for v in paulis]
assert rank==[1,1] and set(normalized_pauli_ranks)=={8}
qpair=abs(ks[[0,7]].conj()@ks[[0,7]].T)**2
dpair=abs(ks[[0,7]])**2@abs(ks[[0,7]].T)**2
sq=np.sqrt(mu)[:,None]*qpair*np.sqrt(mu)[None,:]
sd=np.sqrt(mu)[:,None]*dpair*np.sqrt(mu)[None,:]
commutator=float(np.linalg.norm(sq@sd-sd@sq))
assert commutator>1e-3

out=dict(status='PASS_SOURCE_TRANSFER_BOUNDARIES',protocol_sha256=sha((r/'numerical-protocol.json').read_bytes()),
    audit_code_sha256=sha(Path(__file__).read_bytes()),native_source_sha256=p['native_source_sha256'],
    numpy_version=np.__version__,torch_version=torch.__version__,
    keys=len(ks),queries=len(qs),global_recovery_error=global_error,dephased_recovery_error=dephased_error,
    reduced_state_errors=errors,x_basis_state_error=basis_error,
    shells={k:dict(values=v.tolist(),gram=shellgram[k].tolist(),gram_min_eigenvalue=float(np.linalg.eigvalsh(shellgram[k]).min()),
       cross_min=float(v.min()),cross_negative_entries=int(np.sum(v< -1e-12)),Pauli_directions=int(np.sum(weight==int(k)))) for k,v in shell.items()},
    reduced_kernels=reduced,
    negative_mass_witness=dict(native_legal_angles=ka[[0,7]].tolist(),H1_kernel=h1.tolist(),raw=raw.tolist(),alpha=alpha.tolist(),
       utilities=u.tolist(),logit=bad_logit,native_global_logit=float(original_alpha@u),
       source_kernel_valid_PSD=True,native_nonnegative_attention_contract=False),
    mercer_witness=dict(probabilities=mu.tolist(),positive_eigenvalues=eigenvalues[-2:].tolist(),eigenoperator_residuals=residuals,
       eigenoperator_ranks=rank,all_normalized_Pauli_ranks=sorted(set(normalized_pauli_ranks)),
       eigenoperator_spectra=[np.linalg.eigvalsh(v).tolist() for v in pair],
       Q_kernel=qpair.tolist(),D_kernel=dpair.tolist(),weighted_kernel_commutator_frobenius=commutator,
       conclusion='General Hilbert-Schmidt eigenbasis rotation need not be one physical conjugation of the normalized Pauli basis; arbitrary Pauli weights need not preserve Mercer eigenfunctions.'),
    inputs=dict(key_angles=ka.tolist(),query_angles=qa.tolist()),
    task_rows=0,trained_head_forwards=0,synthetic_full_head_forwards=0,synthetic_circuit_states=len(ks)+len(qs),
    optimizer_updates=0,gpu_allocations=0,server_writes=0,model_api_calls=0,
    new_method_selected=False,new_theorem_claimed=False,goal='ACTIVE',elapsed_seconds=time.monotonic()-start)
for x in out['shells'].values():assert x['gram_min_eigenvalue']>-p['tolerance']
(r/'audit-results.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],global_error=global_error,reduced_error=max(errors.values()),negative_H1=float(h1[1]),
   invalid_attention_logit=bad_logit,Mercer_commutator=commutator,elapsed_seconds=out['elapsed_seconds'])))
