"""Independent dense-gate/Cholesky/basis-free and direct least-squares checks."""
from pathlib import Path
import json,hashlib,itertools,math
import numpy as np

r=Path(__file__).resolve().parent
v=json.loads((r/'audit-results.json').read_bytes());execution=json.loads((r/'execution.json').read_bytes())
assert execution['exit_status']==0 and not execution['literal_stderr']
assert execution['script_sha256']==hashlib.sha256((r/'audit.py').read_bytes()).hexdigest()
printed=json.loads(execution['literal_stdout'])
assert all(printed[k]==v[k] for k in printed)
native=Path(execution['argv'][execution['argv'].index('--native-source')+1])
assert hashlib.sha256(native.read_bytes()).hexdigest()==v['native_source_sha256']

def dense_state(angles):
    state=np.eye(8,dtype=complex)[:,0]
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire]
            ry=np.array([[math.cos(y/2),-math.sin(y/2)],[math.sin(y/2),math.cos(y/2)]])
            rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])
            gate=np.kron(np.kron(np.eye(2**wire),rz@ry),np.eye(2**(2-wire)))
            state=gate@state
        for control,target in [(0,1),(1,2),(2,0)]:
            gate=np.zeros((8,8))
            for col in range(8):
                row=col^((1<<(2-target)) if (col>>(2-control))&1 else 0)
                gate[row,col]=1
            state=gate@state
    return state

inputs=v['synthetic_inputs']
queries=np.array([dense_state(np.array(x)) for x in inputs['effect_angles']])
keys=np.array([dense_state(np.array(x)) for x in inputs['key_angles']])
unpack=lambda x:np.array(x['real'])+1j*np.array(x['imag'])
H=unpack(inputs['null_operator']);plus=unpack(inputs['rho_plus']);minus=unpack(inputs['rho_minus'])
E=np.array([np.outer(x,x.conj()) for x in queries])
assert max(np.max(abs(H-H.conj().T)),abs(np.trace(H)))<1e-12
assert np.max(abs(plus-(np.eye(8)/8+H/16)))<1e-13
assert np.max(abs(minus-(np.eye(8)/8-H/16)))<1e-13
collision=float(np.max(abs(np.array([np.trace((plus-minus)@e) for e in E]))))
assert collision<1e-13 and min(np.linalg.eigvalsh(plus).min(),np.linalg.eigvalsh(minus).min())>.06
gram=abs(queries.conj()@queries.T)**2-1/8
rank=int(np.sum(np.linalg.eigvalsh(gram)>1e-10))
assert rank==40 and 63-rank==23

# Cholesky gauge of the same normalized POVM, not the primary eigensqrt route.
B=np.array([np.outer(x,x.conj())+1e-6*np.eye(8) for x in keys])
C=B.sum(0);L=np.linalg.cholesky(C);Linv=np.linalg.inv(L)
F=np.array([Linv@b@Linv.conj().T for b in B])
chi=(L.conj().T@queries.T).T
chi/=np.linalg.norm(chi,axis=-1,keepdims=True)
alpha=np.array([[float((np.vdot(q,b@q)/np.vdot(q,C@q)).real) for b in B] for q in queries])
conditional=np.array([[float(np.vdot(q,f@q).real) for f in F] for q in chi])
alpha_error=float(np.max(abs(alpha-conditional)))
u=np.sin(np.arange(1,41)[:,None]*np.arange(1,11)[None,:])
z=np.array([np.vdot(q,np.sum(row[:,None,None]*F,axis=0)@q).real for q,row in zip(chi,u)])
readout_error=float(np.max(abs(z-(alpha*u).sum(-1))))
assert alpha_error<1e-12 and readout_error<1e-12

I=np.eye(2);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.])
basis=np.array([I,X,Y,Z])/math.sqrt(2)
projections=np.array([(I+s*P)/2 for P in [X,Y,Z] for s in [-1,1]])
design=np.einsum('aij,bji->ab',projections,basis).real
hk=design.T@design/3
assert np.linalg.eigvalsh(hk).min()>.33
hk_inv=np.linalg.inv(hk);ivec=np.array([math.sqrt(2),0,0,0])
cstar=np.eye(4)-np.outer(ivec,hk_inv@ivec)/(ivec@hk_inv@ivec)
ak=hk_inv@cstar
rho=np.diag([.999,.001]);S=-10*Z
p=np.array([np.trace(rho@a).real for a in projections])
sk=np.zeros((4,4))
for i in range(3):
    rows=design[2*i:2*i+2];prob=p[2*i:2*i+2]
    mean=prob@rows
    sk+=(rows.T*prob)@rows-np.outer(mean,mean)
sk/=9
svec=np.einsum('ij,bji->b',S,basis).real
source_M=float(svec@ak@sk@ak@svec)
mean=variance=tail=0.;observed=[]
for outcomes in itertools.product([-1,1],repeat=3):
    freqs=np.zeros(6);prob=1.
    for i,s in enumerate(outcomes):
        index=2*i+(s==1);freqs[index]=1;prob*=p[index]
    estimate=np.linalg.lstsq(design,freqs,rcond=None)[0]
    rh=np.einsum('b,bij->ij',estimate,basis)
    contrast=float(np.trace((rh-rho)@S).real)
    observed.append(contrast);tail+=prob*(contrast>10);mean+=prob*contrast;variance+=prob*contrast**2
assert abs(tail-.001)<1e-14 and abs(mean)<1e-12
assert abs(variance-.3996)<1e-12 and abs(source_M-variance)<1e-12
h=lambda x:(1+x)*math.log1p(x)-x
bad=math.exp(-source_M*h(10/source_M))
assert bad<5e-11 and tail>10**7*bad
assert abs(bad-v['printed_bound']['printed_upper_bound'])<1e-18
b=max(observed);good=math.exp(-source_M/b**2*h(b*10/source_M))
assert tail<=good<.074

receipt=json.loads((r/'fulltexts/7ccfeee51fabdeb8/receipt.json').read_bytes())
assert receipt['identity']=='verified' and receipt['observed_version']=='2605.25146v1'
assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert len(pages)==46 and '2605.25146v1' in pages[0]['text']
assert 'Theorem 4.25' in pages[20]['text'] and 'Proof of Theorem 4.25' in pages[44]['text']
out=dict(status='PASS_INDEPENDENT_NUMERICAL_AUDIT',
    dense_gate_vs_native_collision_error=collision,basis_free_effect_rank=rank,mixed_state_nullity=23,
    cholesky_conditional_alpha_error=alpha_error,cholesky_signed_readout_error=readout_error,
    direct_lse_exact_enumeration_tail=tail,source_operator_formula_M=source_M,
    enumerated_variance=variance,printed_bound=bad,known_range_corrected_bound=good,
    printed_Theorem4_25_literal_scope_counterexample_verified=True,
    full_source_theorem_suite_verified=False,independent_external_review=False,
    native_source_unchanged=True,task_rows=0,new_updates=0,new_gpu_allocations=0,
    new_T='NOT_ESTABLISHED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',goal='ACTIVE')
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out))
