"""Synthetic native learned-angle permutation audit; no task data or update."""
from pathlib import Path
import sys, types, importlib.util, json, hashlib, math, copy, time
import numpy as np
import torch
torch.set_num_threads(1)
r=Path(__file__).resolve().parent
source=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred')
sha=lambda b:hashlib.sha256(b).hexdigest()
expected={'rmea.py':'9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0',
          'frozen_feature_head.py':'f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9'}
for name,s in expected.items(): assert sha((source/name).read_bytes())==s
pkg=types.ModuleType('native_assignment_port');pkg.__path__=[str(source)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.rmea',source/'rmea.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
protocol=json.loads((r/'synthetic-protocol.json').read_bytes())
torch.manual_seed(protocol['seed'])
p=torch.tensor(protocol['permutation']);inv=torch.argsort(p)
assert sorted(p.tolist())==list(range(12)) and not torch.equal(p,torch.arange(12))
dt=torch.float64;start=time.monotonic()
h=(.2*torch.sin(torch.arange(3*7*1024,dtype=dt)*.017)).reshape(3,7,1024).requires_grad_()
roles=torch.zeros(3,7,4,dtype=dt);roles[:,0,0]=1;roles[:,1,1]=1;roles[:,0,2]=1;roles[:,1,3]=1
mask=torch.arange(7)[None,:]<torch.tensor(protocol['mask_lengths'])[:,None]
coeff=torch.sin(torch.arange(1,121,dtype=dt)).reshape(3,40)
objective=lambda z:(z*coeff).sum()/120+.2*z.square().mean()
state_sha=lambda m:sha(b''.join(k.encode()+v.detach().numpy().tobytes() for k,v in m.state_dict().items()))

def assigned(model):
    flat=math.pi*torch.tanh(model.prepare(torch.cat((h,roles),-1)))
    angles=flat[...,p].reshape(3,7,2,3,2)
    qa=math.pi*torch.tanh(model.effect)
    psi=native.circuit(angles);phi=native.circuit(qa)
    if model.mode=='quantum':prob=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
    else:prob=torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
    raw=(prob+native.EPS)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
    u=torch.einsum('btd,rd->brt',model.value(h),model.decoder)
    z=(alpha*u).sum(-1)+model.bias
    return dict(angles=angles,probability=prob,alpha=alpha,logits=z)

results={}
for mode in protocol['modes']:
    baseline=native.RMEAHead(mode).double();mapped=copy.deepcopy(baseline)
    with torch.no_grad():
        mapped.prepare.weight.copy_(baseline.prepare.weight[inv]);mapped.prepare.bias.copy_(baseline.prepare.bias[inv])
    before=[state_sha(baseline),state_sha(mapped)]
    a=baseline(h,roles,mask,details=True);b=assigned(mapped)
    ga=torch.autograd.grad(objective(a['logits']),tuple(baseline.parameters())+(h,))
    gb=torch.autograd.grad(objective(b['logits']),tuple(mapped.parameters())+(h,))
    names=[n for n,_ in baseline.named_parameters()]+['synthetic_h']
    grad={}
    for name,x,y in zip(names,ga,gb):
        # Mapped parameters are inv-permuted; restore baseline row coordinates.
        if name in ['prepare.weight','prepare.bias']:y=y[p]
        grad[name]=float((x-y).abs().max())
    errors={key:float((a[key]-b[key]).abs().max().detach()) for key in b}
    # An uncompensated routing change is generally different at this fixed checkpoint.
    with torch.no_grad():
        unadjusted_angles=(math.pi*torch.tanh(baseline.prepare(torch.cat((h,roles),-1))))[...,p].reshape(3,7,2,3,2)
        nontrivial=float((native.circuit(unadjusted_angles)-a['psi']).abs().max())
    assert max(errors.values())<protocol['tolerance'] and max(grad.values())<protocol['tolerance']
    assert nontrivial>1e-4 and [state_sha(baseline),state_sha(mapped)]==before
    assert sum(x.numel() for x in baseline.parameters())==protocol['registered_parameters']
    assert len(list(baseline.parameters()))==protocol['parameter_tensors']
    results[mode]=dict(forward_errors=errors,gradient_errors_in_baseline_coordinates=grad,
       uncompensated_state_change=nontrivial,parameter_hashes_unchanged=True,parameter_state_sha256=before,
       baseline_objective=float(objective(a['logits']).detach()),mapped_objective=float(objective(b['logits']).detach()),
       baseline_logits=a['logits'].detach().tolist(),mapped_logits=b['logits'].detach().tolist())
out=dict(status='PASS_ASSIGNMENT_ONLY_REPARAMETRIZATION',results=results,
    protocol_sha256=sha((r/'synthetic-protocol.json').read_bytes()),code_sha256=sha(Path(__file__).read_bytes()),
    source_sha256=expected,torch_version=torch.__version__,numpy_version=np.__version__,
    synthetic_native_head_forwards=2,synthetic_local_adapter_forwards=2,synthetic_uncompensated_circuit_ports=2,
    task_rows=0,trained_head_forwards=0,optimizer_updates=0,gpu_allocations=0,server_writes=0,model_api_calls=0,
    external_scientific_review=False,independent_task_confirmation=False,
    conclusion='Fixed angle-slot assignment is absorbed by inverse row permutation of the free native affine preparation; known parameter symmetry, not a new mechanism/theorem or a result about raw ordered input grouping.',
    elapsed_seconds=time.monotonic()-start)
(r/'assignment-results.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],forward_max=max(v for x in results.values() for v in x['forward_errors'].values()),gradient_max=max(v for x in results.values() for v in x['gradient_errors_in_baseline_coordinates'].values()),elapsed_seconds=out['elapsed_seconds'])))
