"""Independent dense gates, bitwise Pauli expectations and exact rational witnesses."""
from pathlib import Path
from fractions import Fraction as F
import json,math,itertools,hashlib,time
import numpy as np
r=Path(__file__).resolve().parent;start=time.monotonic()
load=lambda n:json.loads((r/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
a=load('audit-results.json');p=load('numerical-protocol.json');e=load('execution.json')
assert a['status']=='PASS_SOURCE_TRANSFER_BOUNDARIES' and e['exit_status']==0 and e['stderr']==''
assert sha((r/'audit.py').read_bytes())==a['audit_code_sha256']
assert sha((r/'numerical-protocol.json').read_bytes())==a['protocol_sha256']==e['inputs']['numerical-protocol.json']
assert sha((r/'audit-results.json').read_bytes())==e['output_sha256']
assert json.loads(e['stdout'])['status']==a['status']

def circuit(angles):
    v=np.eye(8,dtype=complex)[:,0]
    for layer in angles:
        for wire,(y,z) in enumerate(layer):
            ry=np.array([[np.cos(y/2),-np.sin(y/2)],[np.sin(y/2),np.cos(y/2)]])
            rz=np.diag(np.exp(np.array([-1j,1j])*z/2))
            gates=[np.eye(2),np.eye(2),np.eye(2)];gates[wire]=rz@ry
            v=np.kron(np.kron(gates[0],gates[1]),gates[2])@v
        for control,target in [(0,1),(1,2),(2,0)]:
            mat=np.zeros((8,8))
            for b in range(8):mat[b ^ ((1<<(2-target)) if (b>>(2-control))&1 else 0),b]=1
            v=mat@v
    return v

strings=list(itertools.product(range(4),repeat=3));weights=np.count_nonzero(strings,axis=1)
def expectations(v):
    out=[]
    for string in strings:
        value=0j
        for b in range(8):
            c=b;phase=1
            for j,gate in enumerate(string):
                bit=(b>>(2-j))&1
                if gate in [1,2]:c^=1<<(2-j)
                if gate==2:phase*=1j*(-1)**bit
                if gate==3:phase*=(-1)**bit
            value+=v[c].conjugate()*phase*v[b]
        assert abs(value.imag)<1e-12;out.append(value.real)
    return out

keys=np.array([circuit(x) for x in a['inputs']['key_angles']]);queries=np.array([circuit(x) for x in a['inputs']['query_angles']])
kr=np.array([expectations(v) for v in keys]);qr=np.array([expectations(v) for v in queries])
errors=[]
for H in range(4):
    values=kr[:,weights==H]@qr[:,weights==H].T/math.sqrt(math.comb(3,H)*3**H)
    gram=kr[:,weights==H]@kr[:,weights==H].T/math.sqrt(math.comb(3,H)*3**H)
    errors.extend([float(abs(values-np.array(a['shells'][str(H)]['values'])).max()),float(abs(gram-np.array(a['shells'][str(H)]['gram'])).max())])

def partial_trace(v,keep):
    out=np.zeros((2**len(keep),2**len(keep)),complex)
    bits=lambda n: [(n>>2)&1,(n>>1)&1,n&1]
    index=lambda b:sum(b[j]<<(len(b)-j-1) for j in range(len(b)))
    for i,j in itertools.product(range(8),repeat=2):
        bi,bj=bits(i),bits(j)
        if all(bi[k]==bj[k] for k in range(3) if k not in keep):
            out[index([bi[k] for k in keep]),index([bj[k] for k in keep])]+=v[i]*v[j].conjugate()
    return out

for S in [1,2,3]:
    total=np.zeros((len(keys),len(queries)))
    for keep in itertools.combinations(range(3),S):
        rs=[partial_trace(v,keep) for v in keys];ss=[partial_trace(v,keep) for v in queries]
        total+=np.array([[np.trace(x@y).real for y in ss] for x in rs])/math.comb(3,S)
    errors.append(float(abs(total-np.array(a['reduced_kernels'][str(S)]['mean_hs'])).max()))
assert max(errors)<p['tolerance']

# Literal ideal H1 attention failure, without floating cancellation or source circuit reuse.
eps=F(1,10**6);raw=[1+eps,-1+eps];alpha=[x/sum(raw) for x in raw]
exact_logit=alpha[0]-alpha[1]
assert exact_logit==10**6 and alpha[1]<0
assert abs(a['negative_mass_witness']['logit']-float(exact_logit))<1e-3
rho0=[[F(1,8) for _ in range(8)] for _ in range(8)]
rho1=[[F((-1)**(i.bit_count()+j.bit_count()),8) for j in range(8)] for i in range(8)]
inner=lambda x,y:sum(x[i][j]*y[j][i] for i,j in itertools.product(range(8),repeat=2))
assert inner(rho0,rho0)==inner(rho1,rho1)==1 and inner(rho0,rho1)==0
mu=[F(1,3),F(2,3)]
for rho,w in zip([rho0,rho1],mu):
    op=[[sum(m*v[i][j]*inner(v,rho) for m,v in zip(mu,[rho0,rho1])) for j in range(8)] for i in range(8)]
    assert all(op[i][j]==w*rho[i][j] for i,j in itertools.product(range(8),repeat=2))
comm_squared=2*(mu[0]-mu[1])**2*mu[0]*mu[1]/64
assert comm_squared==F(1,1296)
assert abs(a['mercer_witness']['weighted_kernel_commutator_frobenius']**2-float(comm_squared))<1e-14
assert a['mercer_witness']['eigenoperator_ranks']==[1,1] and a['mercer_witness']['all_normalized_Pauli_ranks']==[8]
assert all(np.linalg.matrix_rank(x,tol=1e-10)==1 for x in [np.array(rho0,float),np.array(rho1,float)])
assert abs(a['reduced_kernels']['1']['sourceS_Pauli_weight_l2_squared']-1.5)<1e-12
assert abs(a['reduced_kernels']['2']['sourceS_Pauli_weight_l2_squared']-1.5)<1e-12
receipt=load('fulltexts/a23ea1b3b66a67ea/receipt.json')
assert sha(Path(receipt['path']).read_bytes())=='c96126385a64b9352f1184aa2557f7313815529d96ad942f6073849b33b204bf'
assert receipt['identity']=='verified'
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert 'U0' in pages[6]['text'] and 'basis trans-' in pages[6]['text']
assert 'Λ' in pages[19]['text'] and 'fixed' in pages[19]['text']
out=dict(status='PASS_INDEPENDENT_DENSE_BITWISE_RATIONAL_AUDIT',max_port_error=max(errors),
    exact_H1_logit=str(exact_logit),exact_alpha=[str(x) for x in alpha],
    exact_Mercer_commutator_squared=str(comm_squared),Mercer_eigenoperator_rank=1,normalized_Pauli_rank=8,
    independent_implementation='NumPy dense8x8 gates, bitwise Pauli expectation, explicit-index partialtrace, Fraction witnesses; noTorch/native import.',
    external_scientific_review=False,independent_task_confirmation=False,new_method_selected=False,new_theorem_claimed=False,
    task_rows=0,trained_head_forwards=0,new_updates=0,new_gpu_allocations=0,goal='ACTIVE',L3='NOT_ACCEPTED',L4T='NOT_ACCEPTED',
    source_pdf_identity_verified=True,input_code_protocol_and_literal_exit_verified=True,
    elapsed_seconds=time.monotonic()-start,
    inputs_sha256={n:sha((r/n).read_bytes()) for n in ['scope.json','numerical-protocol.json','audit.py','audit-results.json','execution.json']},
    verification_code_sha256=sha(Path(__file__).read_bytes()))
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],max_port_error=max(errors),exact_H1_logit=str(exact_logit),exact_commutator_squared=str(comm_squared),elapsed_seconds=out['elapsed_seconds'])))
