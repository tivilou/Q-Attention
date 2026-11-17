"""Descriptive geometry and finite-precision-aware fixed-utility certificates."""
import numpy as np


def tv_rows(a):
    return np.abs(a[:, None, :] - a[None, :, :]).sum(-1) / 2


def fidelity(states):
    result = np.abs(states.conj() @ states.T) ** 2
    assert result.min() >= -1e-12 and result.max() <= 1 + 1e-10
    return np.clip(result, 0, 1)


def ce(z, y):
    peak = z.max()
    return float(peak + np.log(np.exp(z - peak).sum()) - z[y])


def analyze(phi, psi, alpha, utilities, bias, native_logits, target, eps=1e-6):
    """One unpadded source example. No target enters the intervention.

    Native alpha is retained for readout. Normalized copies define probability
    geometry. A midpoint/mass correction makes signed utility bounds valid even
    when native float32 attention row masses differ slightly from one.
    """
    phi = np.asarray(phi, np.complex128)
    psi = np.asarray(psi, np.complex128)
    a = np.asarray(alpha, np.float64)
    u = np.asarray(utilities, np.float64)
    b = np.asarray(bias, np.float64)
    native = np.asarray(native_logits, np.float64)
    assert phi.shape == (40, 8) and psi.ndim == 2 and psi.shape[1] == 8
    assert a.shape == u.shape == (40, len(psi)) and len(psi) > 0
    assert b.shape == native.shape == (40,) and 0 <= target < 40
    assert all(np.isfinite(x).all() for x in [phi, psi, a, u, b, native])
    assert (a >= 0).all() and (a.sum(-1) > 0).all()
    norms_phi = np.linalg.norm(phi, axis=1)
    norms_psi = np.linalg.norm(psi, axis=1)
    norm_residual = float(max(abs(norms_phi - 1).max(), abs(norms_psi - 1).max()))
    assert norm_residual <= 5e-6, 'Unexpected native state normalization error'
    phi = phi / norms_phi[:, None]
    psi = psi / norms_psi[:, None]
    B = psi.T @ psi.conj() + len(psi) * eps * np.eye(8)
    gram = phi.conj() @ B @ phi.T
    diagonal = np.real(np.diag(gram))
    assert diagonal.min() > 0
    effective = np.abs(gram) ** 2 / np.outer(diagonal, diagonal)
    assert effective.min() >= -1e-12 and effective.max() <= 1 + 1e-9
    effective = np.clip(effective, 0, 1)
    ideal = np.abs(phi.conj() @ psi.T) ** 2 + eps
    ideal /= ideal.sum(-1, keepdims=True)
    normalized = a / a.sum(-1, keepdims=True)
    residual = np.abs(normalized - ideal).sum(-1) / 2
    actual_tv = tv_rows(normalized)
    geometric_bound = np.sqrt(1 - effective) + residual[:, None] + residual[None, :]
    geometry_violation = float(max(0, (actual_tv - geometric_bound).max()))
    assert geometry_violation <= 1e-8
    pi = (np.arange(40) + 1) % 40
    baseline = (a * u).sum(-1) + b
    changed = (a[pi] * u).sum(-1) + b
    assert np.array_equal(baseline, (a.copy() * u.copy()).sum(-1) + b)
    difference = a[pi] - a
    utility_range = np.ptp(u, axis=1)
    midpoint = (u.max(-1) + u.min(-1)) / 2
    mass_correction = abs(midpoint) * abs(difference.sum(-1))
    delta = utility_range * abs(difference).sum(-1) / 2 + mass_correction
    logit_violation = float(max(0, (abs(changed - baseline) - delta).max()))
    assert logit_violation <= 1e-8 * max(1, abs(u).max())
    winner = int(baseline.argmax())
    other = np.arange(40) != winner
    certified = bool(np.all((baseline[winner] - baseline[other]) > (delta[winner] + delta[other])))
    assert not certified or winner == int(changed.argmax())
    terms = baseline - baseline[target] + delta + delta[target]
    terms[target] = 0
    peak = terms.max()
    upper = float(peak + np.log(np.exp(terms - peak).sum()))
    changed_ce = ce(changed, target)
    assert changed_ce <= upper + 1e-8 * max(1, abs(upper))
    return dict(raw_fidelity=fidelity(phi), effective_fidelity=effective, actual_tv=actual_tv,
                ideal_tv=tv_rows(ideal), geometry_violation=geometry_violation,
                logit_violation=logit_violation, state_norm_residual=norm_residual,
                ideal_native_tv_residual=float(residual.max()),
                native_mass_residual=float(abs(a.sum(-1) - 1).max()),
                mass_correction_max=float(mass_correction.max()),
                reduction_residual=float(abs(baseline - native).max()),
                native_argmax=int(native.argmax()), baseline_argmax=winner,
                changed_argmax=int(changed.argmax()), target=int(target),
                baseline_ce=ce(baseline, target), changed_ce=changed_ce,
                ce_upper=upper, decision_certified=certified,
                cyclic_logit_absolute_change_mean=float(abs(changed - baseline).mean()),
                cyclic_logit_bound_mean=float(delta.mean()))
