"""Synthetic algebra/gradient/model checks; no task rows or efficacy fits."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
sys.path.insert(0, str(ROOT))
from plugin import frame_probability, InvSqrt, MODES, make_head

torch.set_num_threads(1)
torch.manual_seed(13)
errors = []
for complex_mode in (False, True):
    dtype = torch.complex128 if complex_mode else torch.float64
    k = torch.randn(2, 3, 4, dtype=dtype)
    q = torch.randn(7, 4, dtype=dtype)
    k = F.normalize(k, dim=-1).requires_grad_()
    q = F.normalize(q, dim=-1).requires_grad_()
    actual = frame_probability(k, q)
    kk, qq = k.detach().numpy(), q.detach().numpy()
    gram = qq.T @ qq.conj()+7e-6*np.eye(4)
    val, vec = np.linalg.eigh(gram)
    inverse = (vec/np.sqrt(val)) @ vec.conj().T
    effects = np.array([inverse @ (np.outer(x, x.conj())+1e-6*np.eye(4)) @ inverse for x in qq])
    expected = np.einsum('bti,rij,btj->brt', kk.conj(), effects, kk).real
    error = float(np.max(abs(expected-actual.detach().numpy())))
    errors.append(error)
    assert error < 1e-11
    assert float((actual.sum(1)-1).abs().max()) < 1e-11
    assert torch.autograd.gradcheck(frame_probability, (k, q), eps=1e-6, atol=1e-5)
    # Repeated eigenvalues must not introduce eigenvector-gap divisions.
    x = (torch.eye(3, dtype=dtype)*2).requires_grad_()
    assert torch.autograd.gradcheck(InvSqrt.apply, (x,), eps=1e-6, atol=1e-5)

k = F.normalize(torch.randn(2, 3, 4, dtype=torch.complex128), dim=-1).requires_grad_()
q = F.normalize(torch.randn(7, 4, dtype=torch.complex128), dim=-1).requires_grad_()
assert torch.autograd.gradcheck(lambda a,b: frame_probability(a,b,True), (k,q), eps=1e-6, atol=1e-5)
diag_expected = np.einsum('rd,btd->brt',
    (abs(q.detach().numpy())**2+1e-6)/(abs(q.detach().numpy())**2).sum(0).__add__(7e-6),
    abs(k.detach().numpy())**2)
assert np.max(abs(diag_expected-frame_probability(k,q,True).detach().numpy())) < 1e-12

# Regression: rounding squared amplitudes before promotion breaks FP64 frame policy.
k32 = F.normalize(torch.randn(2, 3, 4, dtype=torch.complex64), dim=-1)
q32 = F.normalize(torch.randn(7, 4, dtype=torch.complex64), dim=-1)
reference32 = frame_probability(k32.to(torch.complex128),q32.to(torch.complex128),True).float()
assert torch.equal(frame_probability(k32,q32,True),reference32)
wrong_p, wrong_q = k32.abs().square().double(), q32.abs().square().double()
wrong = torch.einsum('rd,btd->brt',(wrong_q+1e-6)/(wrong_q.sum(0)+7e-6),wrong_p).float()
assert not torch.equal(wrong,reference32)

# A tight frame yields the original EPS-regularized row attention exactly.
k = F.normalize(torch.randn(2, 3, 4, dtype=torch.float64), dim=-1)
q = torch.eye(4, dtype=torch.float64)
raw = torch.einsum('rd,btd->brt', q, k).square()+1e-6
assert torch.allclose(F.normalize(frame_probability(k,q),p=1,dim=-1), F.normalize(raw,p=1,dim=-1),atol=1e-12)

model_records = {}
h = torch.randn(2, 3, 1024, dtype=torch.float64)
roles = torch.randn(2, 3, 4, dtype=torch.float64)
mask = torch.tensor([[True, True, False],[True, True, True]])
for kind in MODES:
    m = make_head(kind).double()
    z = m(h, roles, mask)
    assert z.shape == (2, 40) and z.isfinite().all()
    padded = (F.pad(h,(0,0,0,2),value=.7),F.pad(roles,(0,0,0,2),value=.8),F.pad(mask,(0,2)))
    assert float((z-m(*padded)).abs().max()) < 1e-9
    loss = z.square().mean()
    gradients = torch.autograd.grad(loss, tuple(m.parameters()))
    assert all(x.isfinite().all() for x in gradients)
    query_index = list(dict(m.named_parameters())).index('head.query' if kind.startswith('real') else 'head.effect')
    row_norms = gradients[query_index].reshape(40,-1).norm(dim=-1)
    assert row_norms.isfinite().all() and (row_norms > 1e-12).all()
    model_records[kind] = dict(parameters=sum(p.numel() for p in m.parameters()),
        query_gradient_max=float(gradients[query_index].abs().max()),
        all40_query_gradient_min_norm=float(row_norms.min()))
    assert model_records[kind]['parameters'] == 81028

v = dict(status='PASS_SYNTHETIC_ONLY', numpy_max_error=max(errors),
    real_and_complex_gradcheck=True, repeated_eigenvalue_gradcheck=True, diagonal_gradcheck=True,
    tight_frame_endpoint=True,padding=True,model_records=model_records,
    task_actual8='NOT_RUN_REQUIRES_ALLOCATED_TASK_ACCESS', full_efficacy='NOT_RUN',
    plugin_sha256=hashlib.sha256((ROOT/'plugin.py').read_bytes()).hexdigest())
(ROOT/'plugin-verification.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
print(json.dumps(v))
