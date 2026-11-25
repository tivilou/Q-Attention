"""Native12-angle overlap as a fixed-width real tensor contraction, no task data."""
from pathlib import Path
import sys,types,importlib.util,hashlib,json,math,itertools,time
import numpy as np
import torch
torch.set_num_threads(1)
r=Path(__file__).resolve().parent
root=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred')
sha=lambda b:hashlib.sha256(b).hexdigest()
source=root/'rmea.py';base=root/'frozen_feature_head.py'
assert sha(source.read_bytes())=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
assert sha(base.read_bytes())=='f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9'
pkg=types.ModuleType('qattn_etk_native');pkg.__path__=[str(root)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.rmea',source)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
protocol=json.loads((r/'numerical-protocol.json').read_bytes())
seed=protocol['seed'];rng=np.random.default_rng(seed);torch.manual_seed(seed)
dt=torch.float64;start=time.monotonic()
local=np.array([np.eye(2),[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
strings=list(itertools.product(range(4),repeat=3))
paulis=np.array([np.kron(np.kron(local[a],local[b]),local[c]) for a,b,c in strings])
zindices=np.array([j for j,s in enumerate(strings) if all(k in (0,3) for k in s)])
ring=np.eye(64)
for control,target in [(0,1),(1,2),(2,0)]:
    perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
    U=np.eye(8)[perm]
    transformed=U@paulis@U.T
    R=np.einsum('aij,bji->ab',paulis,transformed).real/8
    assert np.max(abs(R-np.rint(R)))<1e-15 and np.max(abs(R@R.T-np.eye(64)))<1e-15
    ring=R@ring
ring=torch.tensor(ring,dtype=dt)
v0=torch.tensor(np.array([1/math.sqrt(8) if all(k in (0,3) for k in s) else 0 for s in strings]),dtype=dt)

def density_tensor(angles):
    # Each local transfer matrix equals A0 + Ac cos(theta) + As sin(theta).
    v=v0.expand(angles.shape[:-3]+(64,))
    for layer in range(2):
        for wire in range(3):
            for gate in range(2):
                theta=angles[...,layer,wire,gate];c=torch.cos(theta);s=torch.sin(theta)
                one=torch.ones_like(c);zero=torch.zeros_like(c)
                if gate==0:
                    rows=[(one,zero,zero,zero),(zero,c,zero,s),(zero,zero,one,zero),(zero,-s,zero,c)]
                else:
                    rows=[(one,zero,zero,zero),(zero,c,-s,zero),(zero,s,c,zero),(zero,zero,zero,one)]
                M=torch.stack([torch.stack(row,-1) for row in rows],-2)
                shape=v.shape;v=v.reshape(shape[:-1]+(4,4,4)).movedim(-3+wire,-1)
                v=torch.einsum('...ij,...abj->...abi',M,v).movedim(-1,-3+wire).reshape(shape)
        v=v@ring.T
    return v

angles=torch.tensor(rng.uniform(-2.7,2.7,(96,2,3,2)),dtype=dt)
states=native.circuit(angles).detach().numpy()
rho=np.einsum('ni,nj->nij',states,states.conj())
coords=np.einsum('aij,nji->na',paulis,rho).real/math.sqrt(8)
tensorcoords=density_tensor(angles).detach().numpy()
gram=tensorcoords@tensorcoords.T
gd=tensorcoords[:,zindices]@tensorcoords[:,zindices].T
gram_native=abs(states.conj()@states.T)**2
gd_native=abs(states)**2@abs(states.T)**2
geometry=dict(density_coordinate_error=float(np.max(abs(coords-tensorcoords))),
    quantum_gram_error=float(np.max(abs(gram-gram_native))),
    dephased_gram_error=float(np.max(abs(gd-gd_native))),
    quantum_gram_min_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
    dephased_gram_min_eigenvalue=float(np.linalg.eigvalsh(gd).min()),
    coherence_gram_min_eigenvalue=float(np.linalg.eigvalsh(gram-gd).min()),
    quantum_gram_numeric_rank=int(np.linalg.matrix_rank(gram,tol=1e-10)),
    dephased_gram_numeric_rank=int(np.linalg.matrix_rank(gd,tol=1e-10)),
    coherence_offdiagonal_positive=int(np.sum((gram-gd-np.diag(np.diag(gram-gd)))>1e-10)),
    coherence_offdiagonal_negative=int(np.sum((gram-gd-np.diag(np.diag(gram-gd)))< -1e-10)))
assert max(geometry[k] for k in ['density_coordinate_error','quantum_gram_error','dephased_gram_error'])<1e-11
assert min(geometry[k] for k in ['quantum_gram_min_eigenvalue','dephased_gram_min_eigenvalue','coherence_gram_min_eigenvalue'])> -1e-11
assert geometry['quantum_gram_numeric_rank']<=64 and geometry['dephased_gram_numeric_rank']<=8

head=native.RMEAHead('quantum').double()
h=torch.tensor(rng.normal(0,.2,(3,7,1024)),dtype=dt,requires_grad=True)
roles=torch.zeros((3,7,4),dtype=dt);roles[:,0,0]=1;roles[:,1,1]=1;roles[:,0,2]=1;roles[:,1,3]=1
mask=torch.tensor([[1]*7,[1]*5+[0]*2,[1]+[0]*6],dtype=torch.bool)
weights=torch.tensor(np.sin(np.arange(1,121)).reshape(3,40),dtype=dt)
state_hash=lambda model:sha(b''.join(k.encode()+v.detach().numpy().tobytes() for k,v in model.state_dict().items()))
before=state_hash(head)
objective=lambda z:(z*weights).sum()/120+.2*z.square().mean()

def route(prob,u,bias):
    raw=(prob+native.EPS)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
    return alpha,(alpha*u).sum(-1)+bias

def port(ka,qa,u,bias,mode,representation):
    if representation=='native':
        psi=native.circuit(ka);phi=native.circuit(qa)
        p=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square() if mode=='quantum' else torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
    else:
        vk=density_tensor(ka);vq=density_tensor(qa)
        if mode=='dephased':vk=vk[...,zindices];vq=vq[...,zindices]
        p=torch.einsum('rd,btd->brt',vq,vk)
    alpha,z=route(p,u,bias);return p,alpha,z

full_checks={};ports={}
for mode in ['quantum','dephased']:
    model=native.RMEAHead(mode).double();model.load_state_dict(head.state_dict())
    detail=model(h,roles,mask,details=True)
    pp,aa,zz=port(detail['angles'],detail['effect_angles'],detail['utilities'],model.bias,mode,'tensor')
    params=list(model.named_parameters());variables=[p for _,p in params]+[h]
    gn=torch.autograd.grad(objective(detail['logits']),variables,retain_graph=True)
    gt=torch.autograd.grad(objective(zz),variables)
    errors={name:float((a-b).abs().max()) for name,a,b in zip([n for n,_ in params]+['synthetic_h'],gn,gt)}
    checks=dict(probability=float((detail['probability']-pp).abs().max().detach()),
        alpha=float((detail['alpha']-aa).abs().max().detach()),
        logits=float((detail['logits']-zz).abs().max().detach()),
        parameter_and_input_gradient_errors=errors,all_parameter_tensors_nonzero=all(float(g.norm())>0 for g in gn[:-1]),
        registered_parameters=sum(p.numel() for _,p in params),state_hash_unchanged=state_hash(model)==before)
    assert max(checks[k] for k in ['probability','alpha','logits'])<1e-11
    assert max(errors.values())<1e-10 and checks['all_parameter_tensors_nonzero'] and checks['state_hash_unchanged']
    full_checks[mode]=checks
    if mode=='quantum':
        ka=detail['angles'].detach().numpy();qa=detail['effect_angles'].detach().numpy()
        u=detail['utilities'].detach().numpy();bias=model.bias.detach().numpy()

kd=rng.normal(size=ka.shape);kd/=np.linalg.norm(kd)
qd=rng.normal(size=qa.shape);qd/=np.linalg.norm(qd)
for mode in ['quantum','dephased']:
    kt=torch.tensor(ka,dtype=dt,requires_grad=True);qt=torch.tensor(qa,dtype=dt,requires_grad=True)
    ut=torch.tensor(u,dtype=dt);bt=torch.tensor(bias,dtype=dt)
    pn,an,zn=port(kt,qt,ut,bt,mode,'native');pt,at,zt=port(kt,qt,ut,bt,mode,'tensor')
    gn=torch.autograd.grad(objective(zn),(kt,qt),retain_graph=True)
    gt=torch.autograd.grad(objective(zt),(kt,qt))
    derivative=lambda gs:float(sum((g*torch.tensor(direction,dtype=dt)).sum() for g,direction in zip(gs,[kd,qd])))
    ports[mode]=dict(objective=float(objective(zn).detach()),native_directional_derivative=derivative(gn),
        tensor_directional_derivative=derivative(gt),angle_gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(gn,gt)),
        probabilities=pn.detach().numpy().tolist(),alpha=an.detach().numpy().tolist(),logits=zn.detach().numpy().tolist())
    assert ports[mode]['angle_gradient_max_error']<1e-11
assert state_hash(head)==before
out=dict(status='PASS_SYNTHETIC_NATIVE_TENSOR_EQUIVALENCE',
    native_source_sha256=sha(source.read_bytes()),base_source_sha256=sha(base.read_bytes()),
    protocol_sha256=sha((r/'numerical-protocol.json').read_bytes()),
    audit_code_sha256=sha(Path(__file__).read_bytes()),torch_version=torch.__version__,numpy_version=np.__version__,
    geometry=geometry,full_head=full_checks,routing_ports=ports,
    synthetic_inputs=dict(geometry_angles=angles.numpy().tolist(),key_angles=ka.tolist(),query_angles=qa.tolist(),
        utilities=u.tolist(),bias=bias.tolist(),mask=mask.numpy().tolist(),objective_weights=weights.numpy().tolist(),
        key_angle_direction=kd.tolist(),query_angle_direction=qd.tolist()),
    construction=dict(sites=12,local_features=['1','cos(theta)','sin(theta)'],virtual_dimension_upper_bound=64,
        unexpanded_feature_dimension=3**12,core_matrix_rank_upper_bound=64,
        core_mpo_bond_upper_bound=64**2,minimal_bond_dimension_not_computed=True,
        source_normalized_T_rescaling='SourceT=(1,cos,sin)/sqrt2; source-coordinate core=2^12 times our unnormalized-feature core.'),
    task_rows=0,trained_head_forwards=0,synthetic_native_head_forwards=2,optimizer_updates=0,gpu_allocations=0,
    new_method=False,new_T=False,seconds=time.monotonic()-start,
    limitations=['Exact angle-port and synthetic-head equivalence, not natural-data relevance or task utility.',
        'Known PTM/tensor algebra; matrix rank is not minimal MPO bond dimension.',
        'Pointwise learned-parameter substitution holds; source fixed-kernel/Mercer rates do not establish joint training/Macro guarantees.'])
(r/'audit-results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps({k:out[k] for k in ['status','geometry','full_head','construction','task_rows','optimizer_updates','seconds']}))
