"""Local synthetic-only algebra, stop-gradient, native endpoint and hook tests."""
from pathlib import Path
import ast, copy, hashlib, json, sys
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

r = Path(__file__).resolve().parent
sys.path.insert(0, str(r))
sys.path.insert(0, str(r.parent/'qattn-relation-frame-20261125/source'))
from consistency import objective, masked_unit, kl_from_logits, resolve_backward
from plugin import initialize, MODES, backward as new_backward
from baselines.retacred.rmea import RMEAHead

torch.set_num_threads(1)
torch.manual_seed(13)
torch.use_deterministic_algorithms(True)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.double)/3)
        self.calls = 0

    def forward(self, h, roles, mask):
        self.calls += 1
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        return torch.tanh(pooled @ self.weight.T)


def logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


mask = torch.tensor([[True, True, False], [True, True, True]])
h = torch.randn(2, 3, 4, dtype=torch.double)
roles = torch.randn(2, 3, 4, dtype=torch.double)
y = torch.tensor([0, 2])
direction = torch.randn_like(h)
model = Toy()
base_logits = model(h, roles, mask).detach().numpy()
target_log = logsoftmax(base_logits)
loss, record = objective(model, (h, roles, mask), y,
    generator=None, initial_direction=direction)
assert not record['direction'].requires_grad and not record['perturbation'].requires_grad
assert record['power_norm'].min() > 0
assert all(p.grad is None for p in model.parameters()), 'Power step must not accumulate parameter gradients'
assert (record['perturbation'][~mask] == 0).all()
norm = torch.where(mask[..., None], h, 0.).flatten(1).norm(dim=1)
norm_error = float((record['perturbation'].flatten(1).norm(dim=1) - .01*norm).abs().max())
assert norm_error < 1e-14

# Independent NumPy objective with exactly frozen clean target and VAT direction.
perturb = record['perturbation'].numpy()
def numpy_objective(weight):
    def z(features):
        pool = (features*mask.numpy()[..., None]).sum(1)/mask.numpy().sum(1)[:, None]
        return np.tanh(pool @ weight.T)
    clean_log = logsoftmax(z(h.numpy()))
    dirty_log = logsoftmax(z(h.numpy()+perturb))
    ce = -clean_log[np.arange(2), y.numpy()].mean()
    kl = (np.exp(target_log)*(target_log-dirty_log)).sum(-1).mean()
    return ce+kl

value_error = abs(float(loss.detach())-numpy_objective(model.weight.detach().numpy()))
assert value_error < 1e-13
loss.backward()
weight = model.weight.detach().numpy().copy()
numeric = np.zeros_like(weight)
for index in np.ndindex(weight.shape):
    plus, minus = weight.copy(), weight.copy()
    plus[index] += 1e-5
    minus[index] -= 1e-5
    numeric[index] = (numpy_objective(plus)-numpy_objective(minus))/2e-5
gradient_error = float(np.max(np.abs(numeric-model.weight.grad.numpy())))
assert gradient_error < 1e-9

# Independent analytic input gradient at the finite-difference probe, including
# tanh, mean pooling and per-example relative xi, not reusing returned direction.
unit_np = direction.numpy()*mask.numpy()[...,None]
unit_np /= np.linalg.norm(unit_np.reshape(2,-1),axis=1)[:,None,None]
hnorm_np = np.linalg.norm((h.numpy()*mask.numpy()[...,None]).reshape(2,-1),axis=1)
probe_h = h.numpy()+.001*hnorm_np[:,None,None]*unit_np
pooled = (probe_h*mask.numpy()[...,None]).sum(1)/mask.numpy().sum(1)[:,None]
probe_z = np.tanh(pooled @ weight.T)
gz_np = (np.exp(logsoftmax(probe_z))-np.exp(target_log))/2
pool_grad = (gz_np*(1-probe_z**2)) @ weight
gh_np = pool_grad[:,None,:]*mask.numpy()[...,None]/mask.numpy().sum(1)[:,None,None]
probe_grad = .001*hnorm_np[:,None,None]*gh_np
expected_direction = probe_grad/np.linalg.norm(probe_grad.reshape(2,-1),axis=1)[:,None,None]
power_direction_error = float(np.max(np.abs(expected_direction-record['direction'].numpy())))
assert power_direction_error < 1e-9

# KL itself has no target gradient, and the prediction gradient matches p-q.
q = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
z = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
k = kl_from_logits(q, z)
gq, gz = torch.autograd.grad(k, (q,z), allow_unused=True)
assert gq is None
assert torch.allclose(gz, (z.softmax(-1)-q.detach().softmax(-1))/2, atol=1e-15, rtol=1e-13)

# With identical initial direction, random arm really skips the power step.
random_model = copy.deepcopy(model)
random_model.calls = 0
_, rr = objective(random_model, (h,roles,mask), y,
    generator=None, directed=False, initial_direction=direction)
unit, _ = masked_unit(direction, mask)
assert torch.equal(unit, rr['direction']) and random_model.calls == 2
assert float((rr['direction']-record['direction']).abs().max()) > .01
_, zero = objective(Toy(), (h*0,roles,mask), y,
    generator=None, initial_direction=direction*0)
assert zero['initial_zero_norm'] == zero['power_zero_norm'] == 2
assert (zero['perturbation'] == 0).all()

# Padding values are excluded from BOTH h-radius and random-direction norm.
padded_h = F.pad(h, (0,0,0,2), value=23.)
padded_roles = F.pad(roles, (0,0,0,2), value=-17.)
padded_mask = F.pad(mask, (0,2))
padded_direction = F.pad(direction, (0,0,0,2), value=999.)
plain = Toy()
padded = copy.deepcopy(plain)
a, ra = objective(plain,(h,roles,mask),y,generator=None,initial_direction=direction)
b, rb = objective(padded,(padded_h,padded_roles,padded_mask),y,
    generator=None,initial_direction=padded_direction)
a.backward(); b.backward()
assert torch.allclose(a,b,atol=1e-14,rtol=1e-14)
assert torch.allclose(plain.weight.grad,padded.weight.grad,atol=1e-14,rtol=1e-14)

# Execute EXACT pinned dense baseline function, without importing its unrelated framework.
runtime = r.parent/'qattn-generalization-source-20261125/runtime-source/rmea_fair_reference_runtime.py'
runtime_raw = runtime.read_bytes()
assert hashlib.sha256(runtime_raw).hexdigest() == '4d138efe642fd70944c3a5551c6e728530ed6279433822a6c3172770ff0368f5'
tree = ast.parse(runtime_raw)
functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('require','backward')]
namespace = {}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(runtime),'exec'), namespace)
baseline = namespace['backward']
assert resolve_backward(SimpleNamespace(backward=lambda *a:None), baseline) is baseline
try:
    resolve_backward(SimpleNamespace(),baseline,True)
    raise AssertionError('Missing callback accepted')
except ValueError:
    pass
assert resolve_backward(SimpleNamespace(backward=new_backward),baseline,True) is new_backward

# Actual native Q/D/G code, synthetic frozen h only. No task forward/efficacy claim.
torch.manual_seed(13)
initial = RMEAHead('quantum').state_dict()
models = initialize(initial,13)
x = (torch.randn(3,4,1024), torch.randn(3,4,4),
     torch.tensor([[1,1,0,0],[1,1,1,0],[1,1,1,1]],dtype=torch.bool))
labels = torch.tensor([0,17,39])
native_checks = {}
for name, native in models.items():
    assert sum(p.numel() for p in native.parameters()) == 81028
    left, right = copy.deepcopy(native), copy.deepcopy(native)
    before = {k:v.clone() for k,v in native.state_dict().items()}
    gen = torch.Generator().manual_seed(13)
    gen_state = gen.get_state().clone()
    lo = baseline(left,x,labels)
    ro, disabled = objective(right,x,labels,generator=gen,alpha=0)
    ro.backward()
    assert torch.equal(lo,ro) and torch.equal(gen_state,gen.get_state())
    for p,q in zip(left.parameters(),right.parameters()):
        assert torch.equal(p.grad,q.grad)
    ol = torch.optim.AdamW(left.parameters(),lr=.001,weight_decay=.01)
    or_ = torch.optim.AdamW(right.parameters(),lr=.001,weight_decay=.01)
    for m in (left,right):
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    ol.step(); or_.step()
    assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())
    loss = new_backward(native,x,labels)
    assert loss.isfinite() and all(p.grad is not None and p.grad.isfinite().all() for p in native.parameters())
    query = native.head.query if name=='gaussian_vat' else native.head.effect
    row_norms = query.grad.reshape(40,-1).norm(dim=-1)
    assert (row_norms > 0).all()
    assert all(torch.equal(v,native.state_dict()[k]) for k,v in before.items())
    native_checks[name] = dict(parameters=81028,alpha_zero_loss_gradient_update_exact=True,
        rng_unchanged_at_alpha_zero=True,all40_query_grad_min=float(row_norms.min()),
        total_loss=float(loss),components=native.objective_totals)

for name in ('screen.original.py','screen.py','consistency.py','plugin.py','gaussian_reference.py'):
    compile((r/name).read_bytes(),str(r/name),'exec')

result = dict(status='PASS',scope='Local synthetic CPU only; no task samples or GPU allocation',
    numpy_objective_error=value_error,numpy_stopped_gradient_error=gradient_error,
    independent_power_direction_error=power_direction_error,
    perturbation_norm_error=norm_error,target_gradient_none=True,padding_loss_gradient_parity=True,
    power_step_no_parameter_grad_accumulation=True,random_control_skips_power=True,
    zero_norm_branch=True,explicit_callback_disabled_identity=True,native=native_checks,
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [r/'consistency.py',r/'plugin.py',r/'screen.py',r/'gaussian_reference.py',Path(__file__)]})
(r/'synthetic-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
