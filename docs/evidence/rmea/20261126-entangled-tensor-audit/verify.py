"""Independent dense8x8 NumPy gates and finite differences; no Torch/PTM reuse."""
from pathlib import Path
import json,hashlib,math,itertools,time
import numpy as np
r=Path(__file__).resolve().parent
load=lambda n:json.loads((r/n).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
d=load('audit-results.json');p=load('numerical-protocol.json');execution=load('execution.json')
assert execution['exit_status']==0 and execution['literal_stderr']==''
literal=json.loads(execution['literal_stdout'])
assert all(d[k]==v for k,v in literal.items())
assert d['audit_code_sha256']==sha((r/'audit.py').read_bytes())
assert d['protocol_sha256']==sha((r/'numerical-protocol.json').read_bytes())
start=time.monotonic()

def circuit(angles):
    a=np.asarray(angles);flat=a.reshape(-1,2,3,2);out=[]
    for item in flat:
        psi=np.array([1,0,0,0,0,0,0,0],complex)
        for layer in range(2):
            operators=[]
            for wire in range(3):
                y,z=item[layer,wire];c,s=math.cos(y/2),math.sin(y/2)
                Ry=np.array([[c,-s],[s,c]],complex)
                Rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])
                operators.append(Rz@Ry)
            U=np.kron(np.kron(operators[0],operators[1]),operators[2]);psi=U@psi
            for control,target in [(0,1),(1,2),(2,0)]:
                U=np.zeros((8,8))
                for i in range(8):
                    bits=[(i>>2)&1,(i>>1)&1,i&1];bits[target]^=bits[control]
                    dest=bits[0]*4+bits[1]*2+bits[2];U[dest,i]=1
                psi=U@psi
        out.append(psi)
    return np.array(out).reshape(a.shape[:-3]+(8,))

x=d['synthetic_inputs'];ka=np.array(x['key_angles']);qa=np.array(x['query_angles'])
u=np.array(x['utilities']);bias=np.array(x['bias']);mask=np.array(x['mask']);w=np.array(x['objective_weights'])
kd=np.array(x['key_angle_direction']);qd=np.array(x['query_angle_direction'])
def evaluate(keys,queries,mode):
    psi=circuit(keys);phi=circuit(queries)
    if mode=='quantum':prob=abs(np.einsum('rd,btd->brt',phi.conj(),psi))**2
    else:prob=np.einsum('rd,btd->brt',abs(phi)**2,abs(psi)**2)
    raw=(prob+1e-6)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdims=True)
    z=(alpha*u).sum(-1)+bias
    return prob,alpha,z
obj=lambda z:float(np.sum(z*w)/120+.2*np.mean(z*z))
route_errors={};fd={}
for mode in ['quantum','dephased']:
    pp,aa,zz=evaluate(ka,qa,mode);ref=d['routing_ports'][mode]
    errors=dict(probability=float(np.max(abs(pp-ref['probabilities']))),
                alpha=float(np.max(abs(aa-ref['alpha']))),
                logits=float(np.max(abs(zz-ref['logits']))),objective=abs(obj(zz)-ref['objective']))
    assert max(errors.values())<1e-11
    route_errors[mode]=errors;fd[mode]=[]
    for step in p['finite_difference_steps']:
        plus=obj(evaluate(ka+step*kd,qa+step*qd,mode)[2])
        minus=obj(evaluate(ka-step*kd,qa-step*qd,mode)[2])
        estimate=(plus-minus)/(2*step)
        err=abs(estimate-ref['native_directional_derivative'])
        assert err<1e-7
        fd[mode].append(dict(step=step,estimate=estimate,absolute_error=err))
    assert abs(ref['native_directional_derivative']-ref['tensor_directional_derivative'])<1e-11
    assert d['full_head'][mode]['state_hash_unchanged'] and d['full_head'][mode]['registered_parameters']==81028
    assert set(d['full_head'][mode]['parameter_and_input_gradient_errors'])=={'decoder','bias','effect','value.weight','value.bias','prepare.weight','prepare.bias','synthetic_h'}
    assert max(d['full_head'][mode]['parameter_and_input_gradient_errors'].values())<1e-10

states=circuit(np.array(x['geometry_angles']))
qgram=abs(states.conj()@states.T)**2;dgram=abs(states)**2@abs(states.T)**2
ranks=[int(np.linalg.matrix_rank(m,tol=1e-10)) for m in [qgram,dgram]]
assert ranks==[d['geometry']['quantum_gram_numeric_rank'],d['geometry']['dephased_gram_numeric_rank']]==[64,8]
eigenmins=[float(np.linalg.eigvalsh(m).min()) for m in [qgram,dgram,qgram-dgram]]
assert min(eigenmins)> -1e-11
off=qgram-dgram;np.fill_diagonal(off,0)
assert int(np.sum(off>1e-10))==d['geometry']['coherence_offdiagonal_positive']
assert int(np.sum(off< -1e-10))==d['geometry']['coherence_offdiagonal_negative']

receipt=load('fulltexts/e30d554efc54ff02/receipt.json')
assert receipt['identity']=='verified' and receipt['observed_version']=='2503.20683v3'
assert sha(Path(receipt['path']).read_bytes())==receipt['sha256']
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert len(pages)==25 and '2503.20683v3' in pages[0]['text']
assert 'New perspectives' not in pages[0]['text'] and 'Quantum kernels through the lens of entangled tensor kernels' in pages[0]['text']
assert 'Shin' in pages[0]['text'] and 'Sweke' in pages[0]['text'] and 'Jeong' in pages[0]['text']
assert 'Assumption' in pages[15]['text'] and 'Step 1' in pages[21]['text']
out=dict(status='PASS_INDEPENDENT_DENSE_GATE_AND_FINITE_DIFFERENCE_AUDIT',
    routing_errors=route_errors,directional_finite_differences=fd,
    independent_gram_ranks=ranks,independent_gram_min_eigenvalues=eigenmins,
    all7_full_parameter_tensors_and_input_gradient_parity_recorded=True,
    source_pdf_identity_version_and_title_variant_verified=True,
    native_source_unchanged=True,numerical_protocol_and_execution_hashes_verified=True,
    expanded_531441_square_core_materialized=False,minimal_bond_computed=False,
    task_rows=0,trained_head_forwards=0,synthetic_native_head_forwards=2,new_updates=0,new_gpu_allocations=0,
    new_method_selected=False,new_T='NOT_ESTABLISHED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    independent_task_confirmation=False,external_scientific_review=False,goal='ACTIVE',
    independent_cpu_seconds=time.monotonic()-start)
(r/'verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
