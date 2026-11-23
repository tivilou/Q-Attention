"""Label-aware pointwise routing relaxations; no trained-model claim."""
import numpy as np


def intervals(psi, utilities, bias, eps):
    """One unpadded example. Arrays are frozen native float64 tensors.

    Return the generalized-Rayleigh extrema and explicit unit-vector witnesses.
    Floating extrema are numerical estimates, NOT certified outer enclosures.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    u = np.asarray(utilities, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    n, dim = psi.shape
    assert n > 0 and u.shape == (len(bias), n) and eps > 0
    eye = np.eye(dim)
    # rho_t = |psi_t><psi_t|; averaging avoids length-dependent EPS errors.
    s = np.einsum('td,te->de', psi, psi.conj()) / n + eps * eye
    b = np.einsum('rt,td,te->rde', u, psi, psi.conj()) / n
    b += eps * u.mean(-1)[:, None, None] * eye
    se, sv = np.linalg.eigh(s)
    assert se.min() > 0
    w = (sv * se**-.5) @ sv.conj().T
    c = w[None] @ b @ w[None]
    assert np.max(np.abs(c - c.conj().transpose(0, 2, 1))) < 1e-7
    ce, cv = np.linalg.eigh((c + c.conj().transpose(0, 2, 1)) / 2)
    directions = np.stack((cv[..., 0], cv[..., -1]), axis=-1)
    phi = w[None] @ directions
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    probability = np.einsum('rdk,td->rkt', phi.conj(), psi).__abs__()**2
    raw = probability + eps
    replay = (raw * u[:, None]).sum(-1) / raw.sum(-1) + bias[:, None]
    spectral = ce[:, [0, -1]] + bias[:, None]
    convex = np.stack((u.min(-1), u.max(-1)), -1) + bias[:, None]
    return dict(s=s, b=b, spectral=spectral, replay=replay, convex=convex,
                phi=phi, s_eigenvalues=se)


def source_logits(psi, u, phi, bias, eps):
    raw = np.abs(phi.conj() @ psi.T)**2 + eps
    return (raw * u).sum(-1) / raw.sum(-1) + bias


def summarize(z, bounds, replay, convex, labels, calibration, tolerance):
    """Three disjoint statuses. A negative numerical margin is NOT a proof."""
    z, bounds, replay, convex = map(np.asarray, (z, bounds, replay, convex))
    y = np.asarray(labels); tol = np.asarray(tolerance)
    assert z.shape == (len(y), 40) and bounds.shape == replay.shape == convex.shape == z.shape + (2,)
    assert calibration.shape == tol.shape == y.shape
    rows = np.arange(len(y)); others = np.ones(z.shape, dtype=bool); others[rows, y] = False

    def margin(q):
        return q[rows, y, 1] - np.where(others, q[..., 0], -np.inf).max(-1)

    all_margin = margin(bounds); witness_margin = margin(replay)
    fixed_zero = bounds.copy(); fixed_zero[:, 0] = z[:, 0, None]
    fixed_zero_w = replay.copy(); fixed_zero_w[:, 0] = z[:, 0, None]
    fz_margin = margin(fixed_zero); fz_witness = margin(fixed_zero_w)
    native_competitor = np.where(others, z, -np.inf).max(-1)
    true_only_margin = bounds[rows, y, 1] - native_competitor
    true_only_w = replay[rows, y, 1] - native_competitor
    convex_margin = margin(convex)
    pred = z.argmax(-1)  # NumPy matches the source's lowest-index tie rule.
    error = pred != y
    masks = dict(all=np.ones(len(y), bool), calibration=calibration, assessment=~calibration)
    out = {}
    for partition, take in masks.items():
        scopes = dict(all=take, native_errors=take & error,
                      positive_to_zero=take & (y != 0) & (pred == 0),
                      zero_to_positive=take & (y == 0) & (pred != 0),
                      wrong_positive=take & (y != 0) & (pred != 0) & error)
        table = {}
        for name, mask in scopes.items():
            row = dict(n=int(mask.sum()))
            for family, estimate, witness in (
                    ('born_all_rows', all_margin, witness_margin),
                    ('born_class0_fixed', fz_margin, fz_witness),
                    ('born_true_row_only', true_only_margin, true_only_w)):
                feasible = witness > tol
                negative = (estimate < -tol) & ~feasible
                row[family] = dict(explicit_witness_positive=int((mask & feasible).sum()),
                    negative_numerical_margin_not_certificate=int((mask & negative).sum()),
                    unresolved=int((mask & ~feasible & ~negative).sum()))
            row['convex_all_rows'] = dict(positive_margin=int((mask & (convex_margin > tol)).sum()),
                negative_numerical_margin_not_certificate=int((mask & (convex_margin < -tol)).sum()),
                unresolved=int((mask & (np.abs(convex_margin) <= tol)).sum()))
            table[name] = row
        out[partition] = table
    return dict(partitions=out,
        positive_interval_width_mean=float(np.mean(bounds[:, 1:, 1]-bounds[:, 1:, 0])),
        convex_positive_interval_width_mean=float(np.mean(convex[:, 1:, 1]-convex[:, 1:, 0])),
        interpretation='Label-aware per-example independent unit effects; not shared parameters, finite circuit reachability, a Macro ceiling, independent utility, or L5. Negative numerical margins are not exact impossibility certificates.')


def fixtures(eps):
    rng = np.random.default_rng(20261123)
    psi = rng.normal(size=(11, 8)) + 1j*rng.normal(size=(11, 8))
    psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
    phi = rng.normal(size=(40, 8)) + 1j*rng.normal(size=(40, 8))
    phi /= np.linalg.norm(phi, axis=-1, keepdims=True)
    u = rng.normal(size=(40, 11)); bias = rng.normal(size=40)
    q = intervals(psi, u, bias, eps)
    direct = source_logits(psi, u, phi, bias, eps)
    rayleigh = np.einsum('rd,rde,re->r', phi.conj(), q['b'], phi)
    rayleigh /= np.einsum('rd,de,re->r', phi.conj(), q['s'], phi)
    identity = float(np.max(np.abs(direct - bias - rayleigh)))
    witness = float(np.max(np.abs(q['spectral']-q['replay'])))
    assert identity < 1e-10 and witness < 1e-8
    assert np.all(q['spectral'][:, 0] >= q['convex'][:, 0]-1e-8)
    assert np.all(q['spectral'][:, 1] <= q['convex'][:, 1]+1e-8)
    # Identical states: every direction has the uniform-mean utility, even
    # when the arbitrary convex token-attention interval is wide.
    same = np.zeros((3, 8), complex); same[:, 0] = 1
    su = np.tile([-2., 0., 5.], (40, 1)); sb = np.zeros(40)
    constant = intervals(same, su, sb, eps)
    constant_error = float(np.max(np.abs(constant['spectral']-1)))
    assert constant_error < 1e-8
    # Orthogonal states: exact EPS contribution in BOTH S and B is visible.
    eye = np.eye(8, dtype=complex); eu = np.tile(np.arange(8.), (40, 1))
    orthogonal = intervals(eye, eu, sb, eps)
    expected = (np.array([0., 7.]) + 8*eps*3.5)/(1+8*eps)
    orthogonal_error = float(np.max(np.abs(orthogonal['spectral']-expected)))
    assert orthogonal_error < 1e-10
    # Near ties remain unresolved rather than inheriting a lowest-index win.
    zz = np.zeros((1, 40)); bb = np.zeros((1, 40, 2))
    tie = summarize(zz, bb, bb, bb, np.array([1]), np.array([False]), np.array([1e-7]))
    assert tie['partitions']['assessment']['native_errors']['born_all_rows']['unresolved'] == 1
    return dict(random_identity_error=identity, random_extremizer_error=witness,
                identical_state_error=constant_error, orthogonal_eps_error=orthogonal_error,
                near_tie_unresolved=True, status='PASS')
