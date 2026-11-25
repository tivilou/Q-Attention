"""Hermitian coordinates and fixed-cutoff linear-span algebra."""
import math
import numpy as np


def coordinates_torch(psi):
    import torch
    d = psi.shape[-1]
    i, j = torch.triu_indices(d, d, offset=1, device=psi.device)
    off = psi[..., i] * psi[..., j].conj()
    pair = torch.stack((off.real, off.imag), -1).flatten(-2) * math.sqrt(2)
    return torch.cat((psi.abs().square(), pair), -1)


def solve_gram(G, B, rcond=1e-10):
    G = (G + G.T) / 2
    eigenvalues, Q = np.linalg.eigh(G)
    largest = float(eigenvalues[-1])
    assert largest > 0 and float(eigenvalues[0]) >= -largest * 1e-10
    keep = eigenvalues > largest * rcond
    A = Q[:, keep] @ ((Q[:, keep].T @ B) / eigenvalues[keep, None])
    assert np.isfinite(A).all()
    return A, {'rank': int(keep.sum()), 'dimension': len(G), 'relative_cutoff': rcond,
               'eigenvalues': eigenvalues.tolist(), 'absolute_cutoff': largest * rcond}


def observables(A):
    d = math.isqrt(len(A))
    assert d * d == len(A)
    V = np.zeros((A.shape[1], d, d), np.complex128)
    for i in range(d):
        V[:, i, i] = A[i]
    cursor = d
    for i in range(d):
        for j in range(i + 1, d):
            V[:, i, j] = (A[cursor] + 1j * A[cursor + 1]) / math.sqrt(2)
            V[:, j, i] = V[:, i, j].conj()
            cursor += 2
    assert cursor == len(A)
    return V


def weighted_mse(residual, weights, example_count):
    return (weights[:, None] * residual ** 2).sum(0) / example_count
