"""Small dense-operator checks only; no Torch, circuits, task rows or training."""
from pathlib import Path
import hashlib
import json
import ast
import numpy as np

r = Path(__file__).resolve().parent
scope = json.loads((r / 'scope.json').read_bytes())
eps = scope['EPS']
rng = np.random.default_rng(scope['random_seed'])
source = Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
tree = ast.parse(source.read_text('utf-8-sig'))
assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EPS' for t in n.targets)
           and ast.literal_eval(n.value) == eps for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant))


def tr(a):
    x = np.trace(a)
    assert abs(x.imag) < 1e-11
    return float(x.real)


def pure(d):
    x = rng.normal(size=d) + 1j * rng.normal(size=d)
    x /= np.linalg.norm(x)
    return np.outer(x, x.conj())


def mixed(d):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    a = a @ a.conj().T
    return a / np.trace(a)


def observable(d):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    a = (a + a.conj().T) / 2
    return a / np.max(np.abs(np.linalg.eigvalsh(a)))


def real_coordinates(a):
    # Independent coordinate computation, with trace inner product preserved.
    d = len(a)
    return np.array([a[i, i].real for i in range(d)] +
                    [q for i in range(d) for j in range(i + 1, d)
                     for q in (np.sqrt(2) * a[i, j].real, np.sqrt(2) * a[i, j].imag)])


errors = {k: 0. for k in ('poststate_collapse', 'pooled_constant', 'regularized_first_moment',
                         'same_state_two_copy_moment', 'same_state_classical_coordinates', 'probability_coordinates')}
cases = []
for d in scope['dimensions']:
    for kind, count in (('pure', scope['pure_cases_per_dimension']), ('mixed', scope['mixed_cases_per_dimension'])):
        for case in range(count):
            states = [(pure(d) if kind == 'pure' else mixed(d)) for _ in range(scope['tokens_per_case'])]
            E, V = pure(d), observable(d)
            p = np.array([tr(E @ rho) for rho in states])
            n = np.array([float(np.trace(E @ V @ rho).real) for rho in states])
            u = np.array([tr(V @ rho) for rho in states])
            assert p.min() > 0 and p.max() <= 1 + 1e-13
            assert np.max(np.abs(u)) <= 1 + 1e-13
            assert np.max(np.abs(n) - np.sqrt(p)) < 1e-12
            a = (p + eps) / (p + eps).sum()
            post = [E @ rho @ E / pr for rho, pr in zip(states, p)]
            errors['poststate_collapse'] = max(errors['poststate_collapse'], max(float(np.max(np.abs(q - E))) for q in post))
            pooled = sum(w * tr(V @ q) for w, q in zip(a, post))
            errors['pooled_constant'] = max(errors['pooled_constant'], abs(pooled - tr(V @ E)))
            S = sum(states)
            F = (E @ V + V @ E) / 2
            c_direct = float(a @ (n / (p + eps)))
            c_ratio = tr(F @ S) / (tr(E @ S) + len(states) * eps)
            errors['regularized_first_moment'] = max(errors['regularized_first_moment'], abs(c_direct - c_ratio))
            d_direct = float(a @ u)
            M = sum(np.kron(q, q) for q in states)
            d_tensor = (tr(np.kron(E, V) @ M) + eps * tr(V @ S)) / (tr(E @ S) + len(states) * eps)
            errors['same_state_two_copy_moment'] = max(errors['same_state_two_copy_moment'], abs(d_direct - d_tensor))
            X = np.stack([real_coordinates(q) for q in states])
            e, v = real_coordinates(E), real_coordinates(V)
            s, K = X.sum(0), X.T @ X
            classical = (e @ K @ v + eps * v @ s) / (e @ s + len(states) * eps)
            errors['same_state_classical_coordinates'] = max(errors['same_state_classical_coordinates'], abs(d_direct - classical))
            errors['probability_coordinates'] = max(errors['probability_coordinates'], float(np.max(np.abs(X @ e - p))))
            cases.append({'dimension': d, 'state_kind': kind, 'case': case, 'tokens': len(states)})
assert all(x < scope['tolerance']['dense_identity_absolute'] for x in errors.values()), errors

zero = np.array([[1., 0], [0, 0]], complex)
one = np.array([[0., 0], [0, 1]], complex)
plus = np.array([[.5, .5], [.5, .5]], complex)
minus = np.array([[.5, -.5], [-.5, .5]], complex)
Xop = np.array([[0., 1], [1, 0]], complex)


def pooled(E, V, states, route_dephase=False, value_dephase=False):
    probs = [float(np.diag(E).real @ np.diag(q).real) if route_dephase else tr(E @ q) for q in states]
    values = [float(np.diag(V).real @ np.diag(q).real) if value_dephase else tr(V @ q) for q in states]
    a = np.array(probs) + eps
    return float(a @ values / a.sum())


families = {'X_pair': [plus, minus], 'Z_pair': [zero, one]}
contrast = {name: {f'route_{rr}_value_{vv}': pooled(plus, Xop, states, rr == 'D', vv == 'D')
                  for rr in ('Q', 'D') for vv in ('Q', 'D')} for name, states in families.items()}
assert np.array_equal(sum(families['X_pair']), sum(families['Z_pair']))
assert abs(contrast['X_pair']['route_Q_value_Q'] - 1 / (1 + 2 * eps)) < 1e-15
assert all(abs(v) < 1e-15 for k, v in contrast['X_pair'].items() if k != 'route_Q_value_Q')
assert all(abs(v) < 1e-15 for v in contrast['Z_pair'].values())
weak = []
for power in scope['edge_checks']['weak_value_rational_parameter_powers']:
    t = 10. ** -power
    psi = np.array([2 * t, 1 - t * t]) / (1 + t * t)
    rho = np.outer(psi, psi)
    p = tr(zero @ rho)
    n = float(np.trace(zero @ Xop @ rho).real)
    value = (p + eps) * (n / p) / (1 + p + 2 * eps)
    weak.append({'t_power': power, 'p': p, 'n': n, 'native_EPS_pool': value, 'zero_EPS_pool': n / (1 + p)})
assert weak[-1]['native_EPS_pool'] > 100000 and weak[-1]['zero_EPS_pool'] < 1e-10
p_star = eps / (1 + 2 * eps)
regularized_single = np.sqrt(p_star * (1 - p_star)) / (p_star + eps)
assert 499 < regularized_single < 500
out = {'status': 'PASS_DENSE_OPERATOR_IDENTITIES', 'random_cases': len(cases),
       'random_token_density_matrices': sum(c['tokens'] for c in cases), 'case_specification': cases,
       'max_absolute_errors': errors, 'same_density_contrast': contrast, 'weak_value_pole_table': weak,
       'zero_probability_unregularized_weak_value': 'UNDEFINED_NOT_IMPUTED',
       'regularized_single_unit_norm_example': regularized_single,
       'unit_norm_regularized_bound': 1 / (2 * np.sqrt(eps)),
       'native_head_forwards': 0, 'native_circuit_executions': 0, 'task_rows': 0,
       'scope_sha256': hashlib.sha256((r / 'scope.json').read_bytes()).hexdigest(),
       'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
       'new_theory': False, 'new_method_selected': False, 'goal': 'ACTIVE', 'newL3': 'NOT_ACCEPTED', 'newL4T': 'NOT_ACCEPTED'}
(r / 'results.json').write_bytes((json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
print(json.dumps({'status': out['status'], 'random_cases': len(cases), 'random_token_density_matrices': out['random_token_density_matrices'],
                  'max_error': max(errors.values()), 'weak_last_pool': weak[-1]['native_EPS_pool'], 'selected_method': None, 'goal': 'ACTIVE'}))
