"""Decoder decision witnesses; original three-wire factorization, no training."""
import numpy as np


def pt(a, wire):
    """Partial transpose; native wire zero is the most significant bit."""
    return np.asarray(a).reshape((2,) * 6).swapaxes(wire, wire + 3).reshape(8, 8)


def hermitian(a):
    return (a + a.conj().T) / 2


def decision_operator(psi, utilities, bias, rival, sign, mask, epsilon):
    p = np.asarray(psi)[mask]
    c = sign * (np.asarray(utilities)[mask] + bias - rival)
    h = np.einsum('t,ti,tj->ij', c, p, p.conj()) + epsilon * c.sum() * np.eye(8)
    h = hermitian(h)
    norm = float(np.max(np.abs(np.linalg.eigvalsh(h))))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Zero/nonfinite decision operator')
    # Explicit small slack, fixed before seeing any natural decision operator.
    scale = norm * (1 + 1e-10) + 1e-12
    return h, h / scale, scale


def checked_bound(operators, weights, lambdas, qs):
    """Independent float64 feasibility check, not exact interval arithmetic."""
    a = np.asarray(operators, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    raw = np.asarray(lambdas, dtype=np.float64)
    if a.shape != (len(w), 8, 8) or raw.shape != w.shape:
        raise ValueError('Operator/weight shape differs')
    if not np.isfinite(a).all() or not np.isfinite(raw).all():
        raise ValueError('Nonfinite solver values')
    if np.min(w) <= 0 or abs(w.sum() - 1) > 1e-12:
        raise ValueError('Weights do not define a probability distribution')
    norms = np.max(np.abs(np.linalg.eigvalsh(a)), axis=1)
    if np.max(norms) > 1 + 1e-10:
        raise ValueError('Decision operators exceed unit operator norm')
    lam = np.minimum(np.maximum(raw, 0), np.nextafter(w, 0))
    matrix = hermitian(-np.einsum('i,ijk->jk', lam, a))
    projected = []
    for q in qs:
        q = hermitian(np.asarray(q, dtype=np.complex128))
        if q.shape != (8, 8) or not np.isfinite(q).all():
            raise ValueError('Invalid partial-transpose multiplier')
        e, u = np.linalg.eigh(q)
        projected.append(hermitian((u * np.maximum(e, 0)) @ u.conj().T) + 1e-10 * np.eye(8))
    if len(projected) != 3:
        raise ValueError('All three cuts are required')
    scale = 1 + float(np.linalg.norm(matrix, 2)) + max(float(np.linalg.norm(q, 2)) for q in projected)
    allowance = 1e-7 * scale
    cut_bounds = [float(np.linalg.eigvalsh(matrix - pt(q, k))[0]) for k, q in enumerate(projected)]
    delta = min(cut_bounds) - allowance
    q_min = [float(np.linalg.eigvalsh(q)[0]) for q in projected]
    p_min = [float(np.linalg.eigvalsh(matrix - delta * np.eye(8) - pt(q, k))[0]) for k, q in enumerate(projected)]
    if min(q_min) <= 0 or min(p_min) < allowance * .5:
        raise ValueError('Numerical certificate feasibility check failed')
    record = dict(checked_delta=delta, cut_bounds=cut_bounds, numerical_allowance=allowance,
                  q_min_eigenvalues=q_min, p_min_eigenvalues=p_min,
                  max_operator_norm=float(max(norms)), lambda_min=float(lam.min()),
                  lambda_upper_excess=float(np.max(lam - w)), weights_sum=float(w.sum()),
                  scope='Float64 stored-coefficient numerical certificate; not an exact interval proof of the original real-valued circuit.')
    return record, lam, projected


def solve(operators, weights):
    import cvxpy as cp
    n = len(weights)
    lam = cp.Variable(n)
    delta = cp.Variable()
    qs = [cp.Variable((8, 8), hermitian=True) for _ in range(3)]
    matrix = sum(-lam[i] * operators[i] for i in range(n))
    constraints = [lam >= 0, lam <= weights]
    for wire, q in enumerate(qs):
        bit = 1 << (2 - wire)
        transposed = cp.bmat([[q[(i & ~bit) | (j & bit), (j & ~bit) | (i & bit)]
                               for j in range(8)] for i in range(8)])
        constraints.extend((q >> 0, matrix - delta * np.eye(8) - transposed >> 0))
    problem = cp.Problem(cp.Maximize(delta), constraints)
    problem.solve(solver='SCS', eps=1e-7, max_iters=100000, time_limit_secs=300,
                  warm_start=False, verbose=False)
    if lam.value is None or any(q.value is None for q in qs):
        return dict(status='unresolved', solver_status=problem.status), None
    result, clipped, projected = checked_bound(operators, weights, lam.value, [q.value for q in qs])
    result.update(status='complete', solver_status=problem.status, solver_objective=float(problem.value),
                  solver_iterations=problem.solver_stats.num_iters, solver='SCS', cvxpy=cp.__version__)
    return result, dict(lambdas=clipped, qs=np.array(projected), checked_delta=np.array(result['checked_delta']))


def fixtures():
    """Only the new mathematics: masks/normalization, cut order, and witness sign."""
    rng = np.random.default_rng(13)
    psi = rng.normal(size=(7, 8)) + 1j * rng.normal(size=(7, 8))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    mask = np.array([True, True, False, True, False, True, True])
    u = rng.normal(size=7); u[~mask] = 1e12
    h, a, _ = decision_operator(psi, u, .7, -.2, -1, mask, 1e-6)
    error = 0.
    for _ in range(8):
        z = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
        e = z @ z.conj().T; e /= np.trace(e).real
        p = np.einsum('ti,ij,tj->t', psi.conj(), e, psi).real
        alpha = (p + 1e-6) * mask; alpha /= alpha.sum()
        direct = -(float(alpha @ u) + .7 + .2)
        denominator = float(((p + 1e-6) * mask).sum())
        error = max(error, abs(float(np.trace(e @ h).real) / denominator - direct))
    assert error < 1e-10
    z = np.zeros(8, dtype=complex); z[[0, 7]] = 1 / np.sqrt(2)
    rho = np.outer(z, z.conj()); ghz_a = (4 / 3) * rho - np.eye(8)
    qs = [(4 / 3) * (.5 * np.eye(8) - pt(rho, k)) for k in range(3)]
    bound, _, _ = checked_bound(np.array([ghz_a]), np.array([1.]), np.array([1.]), qs)
    assert bound['checked_delta'] > 1 / 3 - 1e-5
    assert all(np.max(np.abs(pt(pt(rho, k), k) - rho)) < 1e-14 for k in range(3))
    bell = np.zeros(8, dtype=complex); bell[[0, 6]] = 1 / np.sqrt(2)
    bell = np.outer(bell, bell.conj())  # Bell on MSB wires0/1; product wire2.
    assert all(abs(np.linalg.eigvalsh(pt(bell, k))[0] + .5) < 1e-12 for k in (0, 1))
    assert np.linalg.eigvalsh(pt(bell, 2))[0] >= -1e-12
    # Product reference: its own positive decision forbids a positive family bound.
    product = np.zeros((8, 8)); product[0, 0] = 1
    product_a = 2 * product - np.eye(8)
    bad, _, _ = checked_bound(np.array([product_a]), np.array([1.]), np.array([1.]), [np.zeros((8, 8))] * 3)
    assert bad['checked_delta'] < 0
    return dict(status='PASS', margin_replay_max_error=error, ghz_checked_delta=bound['checked_delta'],
                product_candidate_delta=bad['checked_delta'], tests=['masked_mixed_effect_margin', 'three_cut_involution', 'MSB_Bell_cut_order', 'GHZ_positive', 'product_rejection'])
