"""Known Hermitian targets, phase-only loss, and unequal sentence lengths."""
from pathlib import Path
import json
import numpy as np
import torch
from span_math import coordinates_torch, solve_gram, observables, weighted_mse

torch.set_num_threads(1)
rng = np.random.default_rng(20261126)
lengths = np.array([2, 5, 1, 7, 4])
w = np.concatenate([np.full(n, 1 / n) for n in lengths])
assert abs(w.sum() - 5) < 1e-14
psi = rng.normal(size=(sum(lengths), 2)) + 1j * rng.normal(size=(sum(lengths), 2))
psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
X = coordinates_torch(torch.from_numpy(psi)).numpy()
V = np.array([[[.3, .4 + .2j], [.4 - .2j, -.1]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
U = np.einsum('ni,cij,nj->nc', psi.conj(), V, psi).real
G, B = X.T @ (w[:, None] * X), X.T @ (w[:, None] * U)
A, info = solve_gram(G, B)
Vhat = observables(A)
matrix_error = float(np.max(np.abs(Vhat - V)))
assert info['rank'] == 4 and matrix_error < 1e-12
test = rng.normal(size=(23, 2)) + 1j * rng.normal(size=(23, 2))
test /= np.linalg.norm(test, axis=-1, keepdims=True)
test_X = coordinates_torch(torch.from_numpy(test)).numpy()
test_U = np.einsum('ni,cij,nj->nc', test.conj(), V, test).real
prediction_error = float(np.max(np.abs(test_X @ A - test_U)))
assert prediction_error < 1e-12
residual = np.arange(len(w) * 3).reshape(len(w), 3) / 10
expected = sum((residual[sum(lengths[:i]):sum(lengths[:i + 1])] ** 2).mean(0) for i in range(5)) / 5
assert np.allclose(weighted_mse(residual, w, 5), expected, atol=1e-14, rtol=0)
assert np.max(np.abs(expected - (residual ** 2).mean(0))) > .1
phase = np.array([[1., 1.], [1., -1.]], complex) / np.sqrt(2)
phase_X = coordinates_torch(torch.from_numpy(phase)).numpy()
target = np.array([[1.], [-1.]])
aq, _ = solve_gram(phase_X.T @ phase_X, phase_X.T @ target)
ad, _ = solve_gram(phase_X[:, :2].T @ phase_X[:, :2], phase_X[:, :2].T @ target)
phase_Q_error = float(np.mean((phase_X @ aq - target) ** 2))
phase_D_error = float(np.mean((phase_X[:, :2] @ ad - target) ** 2))
assert phase_Q_error < 1e-20 and abs(phase_D_error - 1) < 1e-14
out = {'status': 'PASS_KNOWN_OPERATOR_AND_WEIGHTING_CHECKS', 'known_operator_error': matrix_error,
       'unseen_state_prediction_error': prediction_error, 'phase_Q_MSE': phase_Q_error, 'phase_diagonal_MSE': phase_D_error,
       'unequal_sentence_weighting': 'PASS', 'native_head_forwards': 0, 'task_rows': 0}
Path(__file__).with_name('math-check.json').write_bytes((json.dumps(out, indent=2) + '\n').encode())
print(json.dumps(out))
