"""Small deterministic mathematical fixtures; no Torch, data, model or training.

The circuit transcription is checked against the published source algebra.
This is a synthetic scope audit, not reproduction of a trained checkpoint.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
EPS = 1e-6


def circuit(angles):
    state = np.zeros(8, dtype=np.complex128)
    state[0] = 1
    for layer in range(2):
        for wire in range(3):
            y, z = angles[layer, wire]
            c, s = np.cos(y / 2), np.sin(y / 2)
            gate = np.diag([np.exp(-.5j * z), np.exp(.5j * z)]) @ np.array([[c, -s], [s, c]])
            full = np.array([[1.]])
            for w in range(3):
                full = np.kron(full, gate if w == wire else np.eye(2))
            state = full @ state
        for control, target in ((0, 1), (1, 2), (2, 0)):
            ix = [i ^ ((1 << (2 - target)) if (i >> (2 - control)) & 1 else 0) for i in range(8)]
            state = state[ix]
    return state


def basis(d):
    """Real orthonormal Hermitian basis, identity first."""
    out = [np.eye(d) / np.sqrt(d)]
    for k in range(1, d):
        x = np.zeros((d, d), complex)
        x[np.arange(k), np.arange(k)] = 1
        x[k, k] = -k
        out.append(x / np.sqrt(k * (k + 1)))
    for i in range(d):
        for j in range(i + 1, d):
            x = np.zeros((d, d), complex)
            x[i, j] = x[j, i] = 1 / np.sqrt(2)
            out.append(x)
            x = np.zeros((d, d), complex)
            x[i, j], x[j, i] = 1j / np.sqrt(2), -1j / np.sqrt(2)
            out.append(x)
    return np.array(out)


def hs(operators, b):
    return np.einsum('nij,kji->nk', operators, b).real


def main():
    angles = np.zeros((6, 2, 3, 2))
    angles[:, 1, 0, 0] = np.pi * np.array([.5, -.5, .25, -.75, 0, 0])
    psi = np.array([circuit(a) for a in angles])
    rho = np.einsum('ti,tj->tij', psi, psi.conj())
    expected = np.zeros_like(psi)
    expected[:, 0] = np.cos(angles[:, 1, 0, 0] / 2)
    expected[:, 3] = np.sin(angles[:, 1, 0, 0] / 2)
    circuit_residual = float(np.max(np.abs(psi - expected)))
    assert circuit_residual < 1e-14
    p = np.abs(psi[:, 0]) ** 2
    alpha = (p + EPS) / np.sum(p + EPS)
    context = range(4)
    order = sorted(context, key=lambda j: hashlib.sha256(('rmea-context-v1:independent-support-2:' + str(j)).encode()).digest())
    if order == list(context):
        order = order[1:] + order[:1]
    permutation = order + [4, 5]
    assert permutation == [0, 2, 1, 3, 4, 5]
    c = np.array([1, 1, -1, -1, 0, 0])
    witness = float(c @ alpha[permutation])
    matrix_residual = float(np.max(np.abs(np.einsum('t,tij->ij', c, rho))))
    assert matrix_residual < 1e-14 and abs(c @ alpha) < 1e-14
    u = np.array([0, -1, 1, 0, 0, 0])
    # True bias log(39), other 39 logits zero, so true CE=log(1+exp(-alpha.u)).
    actual_ce = float(np.logaddexp(0, -alpha @ u))
    shuffled_ce = float(np.logaddexp(0, -alpha[permutation] @ u))
    assert abs(witness - .17677643013199168) < 1e-14
    assert abs(shuffled_ce - actual_ce - .0883882150659957) < 1e-14

    # Source-compatible informationally complete prepared states: four finite
    # single-wire settings, tensor products, and the same fixed CNOT network.
    # Second-layer rotations are zero. Affine preparation can realize every
    # first-layer tuple via independent h coordinates and inverse tanh.
    one = [(0, 0), (math.pi / 2, 0), (math.pi / 2, math.pi / 2), (-math.pi / 2, 0)]
    prepared = []
    for settings in itertools.product(one, repeat=3):
        a = np.zeros((2, 3, 2))
        a[0] = settings
        prepared.append(circuit(a))
    states = np.einsum('ti,tj->tij', prepared, np.conjugate(prepared))
    b = basis(8)
    B = states + EPS * np.eye(8)
    frame = hs(B, b)
    assert np.linalg.matrix_rank(frame, tol=1e-10) == 64
    # 63 cohort examples share this prepared token multiset but their utilities
    # select different tokens. Values can change independently of preparation.
    effect_angles = np.arange(12, dtype=float).reshape(2, 3, 2) / 13 - .4
    phi = circuit(effect_angles)
    E = np.outer(phi, phi.conj())
    weights = np.einsum('ij,tji->t', E, B).real
    weights /= weights.sum()
    S = B.sum(0)
    M = B[:63] - weights[:63, None, None] * S
    L = hs(M, b[1:])
    singular = np.linalg.svd(L, compute_uv=False)
    assert np.linalg.matrix_rank(L, tol=1e-10) == 63
    assert np.max(np.abs(np.einsum('ij,tji->t', E, M))) < 1e-13

    # One individual scalar logit has a simplex-preserving null direction.
    # The first example's utility selects token 0; move equal mass between1/2.
    delta = min(weights[1], weights[2]) / 2
    alternative = weights.copy()
    alternative[1] += delta
    alternative[2] -= delta
    assert abs(alternative.sum() - 1) < 1e-14
    assert alternative.min() > 0 and alternative[0] == weights[0]

    rng = np.random.default_rng(20261115)
    max_score_identity = 0.
    max_stability_ratio = 0.
    for _ in range(100):
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        v /= np.linalg.norm(v)
        E2 = np.outer(v, v.conj())
        Delta = E2 - E
        denom = np.trace(E2 @ S).real
        scores = np.einsum('ij,tji->t', E2, B).real / denom
        obs = np.einsum('ij,tji->t', Delta, M).real
        max_score_identity = max(max_score_identity, float(np.max(np.abs(scores[:63] - weights[:63] - obs / denom))))
        bound = np.linalg.norm(obs) / singular[-1]
        max_stability_ratio = max(max_stability_ratio, float(np.linalg.norm(Delta) / bound))
        assert np.linalg.norm(Delta) <= bound + 1e-12

    result = dict(
        status='PASS', scope='Synthetic mathematical/code-correspondence fixtures only; not task evidence or L5.',
        six_token=dict(circuit_max_residual=circuit_residual, matrix_identity_max_residual=matrix_residual,
                       permutation=permutation, actual_ce=actual_ce, shuffled_ce=shuffled_ce,
                       ce_damage=shuffled_ce-actual_ce, infeasible_attention_L1_lower_bound=abs(witness)),
        cohort_fixture=dict(tokens_per_example=64, scalar_examples=63, prepared_operator_rank=64,
                            tracezero_constraint_rank=63, least_singular_value=float(singular[-1]),
                            frame_least_singular_value=float(np.linalg.svd(frame,compute_uv=False)[-1]),
                            free_simplex_alternative_L1=float(np.linalg.norm(alternative-weights,1)),
                            random_effect_checks=100, score_identity_max_residual=max_score_identity,
                            maximum_stability_ratio=max_stability_ratio),
        novelty='Known informational-completeness/nullspace and singular-value facts; exact circuit-compatible examples do not make them novel.',
        limits=['No trained effect or task arrays inspected.', 'Full trace-zero rank is sufficient, not necessary for pure/circuit effects.',
                'Individual logit agreement differs from class-label agreement.', 'Mixed-effect feasibility is not circuit reachability.',
                'Fixes preparation, values, decoder, bias, and all other effects; does not establish full-network identifiability.',
                'Off-family replay remains valid for its original functional-dependence question.'])
    (P/'attainability-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
