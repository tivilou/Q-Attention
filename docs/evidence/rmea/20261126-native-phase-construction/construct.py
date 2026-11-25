from pathlib import Path
import hashlib,importlib.util,json,math,sys,time,types
import numpy as np
import torch
from independent_reference import compute as independent_compute

sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent
cfg=json.loads((r/'scope.json').read_bytes())
receipt=json.loads((r/'receipt.before-execution.json').read_bytes())
pins=json.loads((r/'source-pins.json').read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def write(n,x,compact=False):(r/n).write_bytes((json.dumps(x,ensure_ascii=False,allow_nan=False,indent=None if compact else 2,separators=(',',':') if compact else None)+'\n').encode())
assert not (r/'native-stage.json').exists() and not (r/'results.json').exists()
assert sha((r/'scope.json').read_bytes())==receipt['scope_sha256']
assert sha((r/'source-pins.json').read_bytes())==receipt['source_pins_sha256']
for n,h in receipt['code_sha256'].items():assert sha((r/n).read_bytes())==h
for p in pins:assert sha(Path(p['path']).read_bytes())==p['sha256']
torch.set_num_threads(1)
package=types.ModuleType('qattn_native_phase_fixture')
package.__path__=[str(Path(pins[0]['path']).parent)];sys.modules[package.__name__]=package
for filename in ['frozen_feature_head','rmea']:
    name=package.__name__+'.'+filename
    spec=importlib.util.spec_from_file_location(name,Path(package.__path__[0])/(filename+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
native=module
assert native.EPS==cfg['fixed']['EPS'] and native.QUBITS==3 and native.DEPTH==2

def packed(value):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    value=np.asarray(value)
    out=dict(shape=list(value.shape),dtype=str(value.dtype))
    if np.iscomplexobj(value):out.update(real=value.real.tolist(),imag=value.imag.tolist())
    else:out['values']=value.tolist()
    return out

def parameter_hash(head):
    h=hashlib.sha256()
    for name,value in sorted(head.state_dict().items()):
        x=value.detach().contiguous().numpy()
        h.update(name.encode()+b'\0'+str(x.shape).encode()+b'\0'+str(x.dtype).encode()+b'\0'+x.tobytes())
    return h.hexdigest()

counts=dict(head=[],circuit=[])
original_circuit=native.circuit
def counted_circuit(angles):
    counts['circuit'].append(dict(shape=list(angles.shape),dtype=str(angles.dtype),device=str(angles.device)))
    return original_circuit(angles)
native.circuit=counted_circuit
cpu_start=time.process_time();wall_start=time.perf_counter()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(0)
    head=native.RMEAHead('quantum').double().cpu()
with torch.no_grad():
    for p in head.parameters():p.zero_();p.requires_grad_(False)
    head.prepare.weight[0,0]=1
    head.value.weight[0,1]=1;head.value.weight[1,2]=1
    head.decoder[0,0]=1
head.eval()
h=torch.zeros((2,2,1024),dtype=torch.float64,device='cpu')
h[:,:,0]=torch.tensor(cfg['synthetic_input']['h0'],dtype=torch.float64)
h[:,:,1]=torch.tensor(cfg['synthetic_input']['h1'],dtype=torch.float64)
h[:,:,2]=torch.tensor(cfg['synthetic_input']['h2'],dtype=torch.float64)
roles=torch.zeros((2,2,4),dtype=torch.float64)
mask=torch.ones((2,2),dtype=torch.bool)
parameter_before=parameter_hash(head)
handle=head.register_forward_pre_hook(lambda obj,args:counts['head'].append(dict(mode=obj.mode,h_shape=list(args[0].shape))))
try:
    with torch.no_grad():
        quantum=head(h,roles,mask,details=True)
        head.mode='dephased'
        dephased=head(h,roles,mask,details=True)
finally:
    handle.remove();native.circuit=original_circuit
parameter_after=parameter_hash(head)
stage=dict(schema='synthetic-native-stage.v1',scope_sha256=receipt['scope_sha256'],
           inputs={n:packed(x) for n,x in dict(h=h,roles=roles,mask=mask).items()},
           parameter_digest_before=parameter_before,parameter_digest_after=parameter_after,
           parameter_nonzero_counts={n:int(torch.count_nonzero(x)) for n,x in head.state_dict().items()},
           parameter_construction=cfg['parameters'],calls=counts,
           quantum={n:packed(x) for n,x in quantum.items()},dephased={n:packed(x) for n,x in dephased.items()})
write('native-stage.json',stage,compact=True)

with torch.no_grad():
    psi,phi,u=quantum['psi'],quantum['phi'],quantum['utilities']
    rho=torch.einsum('bti,btj->btij',psi,psi.conj())
    bt=rho+native.EPS*torch.eye(8,dtype=torch.complex128)
    B=bt.sum(1)
    A=torch.einsum('bct,btij->bcij',u.to(torch.complex128),bt)
    ph=phi[None,:,:,None]
    denom=(ph.conj().transpose(-1,-2)@B[:,None]@ph)[...,0,0].real
    z0=(ph.conj().transpose(-1,-2)@A@ph)[...,0,0].real/denom
    C=A-z0[...,None,None]*B[:,None]
    q=(torch.matrix_exp(C)@ph)[...,0]
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    qq=q[...,None]
    guided=(qq.conj().transpose(-1,-2)@A@qq)[...,0,0].real/(qq.conj().transpose(-1,-2)@B[:,None]@qq)[...,0,0].real
    guided_probability=torch.einsum('bci,bti->bct',q.conj(),psi).abs().square()
    guided_raw=guided_probability+native.EPS
    guided_alpha=guided_raw/guided_raw.sum(-1,keepdim=True)
    guided_pooled=(guided_alpha*u).sum(-1)
ref=independent_compute(h.numpy(),native.EPS)

def as_np(x):return x.detach().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def error(a,b):return float(np.max(np.abs(as_np(a)-as_np(b))))
expected_psi=np.zeros((2,2,8),dtype=complex)
expected_psi[:,:,0]=1/math.sqrt(2)
expected_psi[:,:,2]=np.array([[1,1],[1,-1]])/math.sqrt(2)
expected_phi=np.zeros((40,8),dtype=complex);expected_phi[:,0]=1
expected_guided=np.zeros((2,40));expected_guided[1,0]=cfg['expected']['guided_class0_B']
errors={
 'native_vs_dense_psi':error(psi,ref['psi']),
 'native_vs_dense_phi':error(phi,ref['phi']),
 'native_vs_analytic_psi':error(psi,expected_psi),
 'native_vs_analytic_phi':error(phi,expected_phi),
 'native_vs_dense_angles':error(quantum['angles'],ref['angles']),
 'state_norm':error(psi.abs().square().sum(-1),1),
 'query_norm':error(phi.abs().square().sum(-1),1),
 'guided_query_norm':error(q.abs().square().sum(-1),1),
 'native_vs_dense_values':error(quantum['values'],ref['values']),
 'native_vs_dense_utilities':error(u,ref['utilities']),
 'native_vs_dense_probability':error(quantum['probability'],ref['probability']),
 'native_probability_vs_half':error(quantum['probability'],.5),
 'native_vs_dense_alpha':error(quantum['alpha'],ref['alpha']),
 'native_vs_original_ratio':error(quantum['logits'],z0),
 'native_vs_dense_original':error(quantum['logits'],ref['original']),
 'native_normalizer':error(quantum['normalizer'][...,0],denom),
 'contexts_same_alpha':error(quantum['alpha'][0],quantum['alpha'][1]),
 'contexts_same_all64values':error(quantum['values'][0],quantum['values'][1]),
 'contexts_same_populations':error(psi[0].abs().square(),psi[1].abs().square()),
 'Q_vs_D_original_logits':error(quantum['logits'],dephased['logits']),
 'Q_vs_D_original_probability':error(quantum['probability'],dephased['probability']),
 'Q_original_zero':error(quantum['logits'],0),
 'D_original_zero':error(dephased['logits'],0),
 'native_vs_dense_A':error(A,ref['A']),
 'native_vs_dense_B':error(B,ref['B']),
 'native_vs_dense_C':error(C,ref['C']),
 'guided_vs_closed_form':error(guided,expected_guided),
 'guided_vs_dense_eigh':error(guided,ref['guided']),
 'guided_query_vs_dense_eigh':error(q,ref['q']),
 'guided_ratio_vs_original_pooling':error(guided,guided_pooled),
 'guided_imaginary_amplitude':error(q.imag,0),
}
cpu_seconds=time.process_time()-cpu_start;wall_seconds=time.perf_counter()-wall_start
tol=cfg['fixed']['tolerance']
gates={name:val<=tol for name,val in errors.items()}
gates.update(source_hashes_unchanged=all(sha(Path(p['path']).read_bytes())==p['sha256'] for p in pins),
             parameters_unchanged=parameter_before==parameter_after,
             exactly_four_nonzero_parameters=sum(stage['parameter_nonzero_counts'].values())==4,
             exact_head_calls=len(counts['head'])==2,
             exact_circuit_calls=len(counts['circuit'])==4,
             CPU_float64=all(x['device']=='cpu' and x['dtype']=='torch.float64' for x in counts['circuit']),
             CPU_time_budget=cpu_seconds<=cfg['budget']['max_internal_CPU_seconds'])
trace=dict(schema='synthetic-native-phase-trace.v1',stage_path='native-stage.json',stage_sha256=sha((r/'native-stage.json').read_bytes()),
           guided={n:packed(x) for n,x in dict(rho=rho,B=B,A=A,z0=z0,C=C,q=q,probability=guided_probability,
                                            alpha=guided_alpha,pooled=guided_pooled,ratio=guided).items()},
           independent={n:packed(x) for n,x in ref.items()},
           closed_form_guided=packed(expected_guided),errors=errors,gates=gates,
           trace_boundary='Actual constructed model simulation; no training, task label, natural feature or checkpoint stage.')
write('trace.json',trace,compact=True)
out=dict(status='PASS_NATIVE_SYNTHETIC_PHASE_CONSTRUCTION' if all(gates.values()) else 'STOP_CONSTRUCTIVE_CLAIM',
         scope_sha256=receipt['scope_sha256'],source_pins_sha256=receipt['source_pins_sha256'],
         native_stage_sha256=sha((r/'native-stage.json').read_bytes()),trace_sha256=sha((r/'trace.json').read_bytes()),
         errors=errors,gates=gates,max_numeric_error=max(errors.values()),CPU_seconds=cpu_seconds,wall_seconds=wall_seconds,
         native_head_calls=len(counts['head']),native_circuit_calls=len(counts['circuit']),
         guided_class0=[float(guided[0,0]),float(guided[1,0])],
         native_basis_support=[0,2],parameter_digest=parameter_before,
         original_all40_logits_zero=True if gates['Q_original_zero'] and gates['D_original_zero'] else False,
         joint_native_reachability='EXPLICIT_SYNTHETIC_FIXTURE_ONLY',
         actually_complex=False,real8_control='This real fixture remains exactly representable; no complex-necessity claim',
         task_utility=False,literature_novelty=False,selected_task_method=False,new_theory=False,
         resources=cfg['resources'],goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
write('results.json',out)
print(json.dumps(out,ensure_ascii=False))
raise SystemExit(0 if all(gates.values()) else 2)
