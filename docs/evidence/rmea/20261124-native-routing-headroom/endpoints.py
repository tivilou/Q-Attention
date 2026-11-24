"""Numerical fixed-key Rayleigh bounds, not a formal interval certificate."""
import numpy as np
import torch

EPS = 1e-6


def bounds(key, utility, bias, mask, query):
    """key[B,T,D], utility[B,R,T]; padded tokens make no contribution."""
    key = key.to(torch.complex128 if key.is_complex() else torch.float64)
    u = utility.double()
    b = bias.double()
    q = query.to(key.dtype)
    m = mask.double()
    d = key.shape[-1]
    eye = torch.eye(d, dtype=key.dtype, device=key.device)
    outer = key[..., :, None] * key[..., None, :].conj()
    B = (outer * m[..., None, None]).sum(1) + EPS * m.sum(1)[:, None, None] * eye
    A = torch.einsum('brt,btij->brij', (u*m[:, None]).to(key.dtype), outer)
    A = A + EPS * (u*m[:, None]).sum(-1)[..., None, None] * eye
    ev, vec = torch.linalg.eigh(B)
    assert bool((ev > 0).all())
    W = (vec * ev.rsqrt()[:, None, :]) @ vec.mH
    C = W[:, None] @ A @ W[:, None]
    hermitian_error = (C-C.mH).abs().amax()
    C = (C+C.mH)*.5
    e, v = torch.linalg.eigh(C)
    low, high = e[..., 0]+b, e[..., -1]+b
    # Check both extremal generalized eigenpairs against the original A,B.
    qext = W[:, None] @ v[..., [0, -1]]
    selected = e[..., [0, -1]]
    residual = A @ qext - (B[:, None] @ qext)*selected[..., None, :]
    scale = ((torch.linalg.matrix_norm(A) + selected.abs().amax(-1)*
              torch.linalg.matrix_norm(B)[:, None])[..., None] *
             torch.linalg.vector_norm(qext, dim=-2)).clamp_min(1e-30)
    relative_residual = (torch.linalg.vector_norm(residual, dim=-2)/scale).amax()
    prob = torch.einsum('rd,btd->brt', q.conj(), key).abs().square()
    raw = (prob+EPS)*m[:, None]
    z = (raw*u).sum(-1)/raw.sum(-1)+b
    outside = torch.maximum(low-z, z-high).clamp_min(0).amax()
    slo = u.masked_fill(~mask[:, None], torch.inf).amin(-1)+b
    shi = u.masked_fill(~mask[:, None], -torch.inf).amax(-1)+b
    nesting = torch.maximum(slo-low, high-shi).clamp_min(0).amax()
    tol = 1e-5*(1+u.masked_fill(~mask[:, None], 0).abs().amax((1, 2))+b.abs().amax())
    diagnostic = dict(minimum_B_eigenvalue=float(ev.amin()),
        maximum_B_condition=float((ev[:, -1]/ev[:, 0]).amax()),
        maximum_hermitian_residual=float(hermitian_error),
        maximum_generalized_eigenpair_relative_residual=float(relative_residual),
        maximum_original_query_outside_bounds=float(outside),
        maximum_query_outside_simplex=float(nesting),
        maximum_query_norm_error=float((q.abs().square().sum(-1)-1).abs().amax()))
    return dict(simplex_low=slo, simplex_high=shi, query_low=low, query_high=high,
                tolerance=tol, query_logits=z, diagnostic=diagnostic)


def numpy_reference(key, utility, bias):
    """Independent scalar loop using Cholesky whitening (no Torch eigensolver)."""
    key = np.asarray(key, dtype=np.complex128 if np.iscomplexobj(key) else np.float64)
    u = np.asarray(utility, dtype=np.float64)
    eye = np.eye(key.shape[-1])
    B = key.T @ key.conj()+EPS*len(key)*eye
    L = np.linalg.cholesky(B)
    low, high = [], []
    for row, b in zip(u, bias):
        A = key.T @ (row[:, None]*key.conj())+EPS*row.sum()*eye
        X = np.linalg.solve(L, A)
        C = np.linalg.solve(L.conj(), X.T).T
        eigen = np.linalg.eigvalsh((C+C.conj().T)/2)
        low.append(eigen[0]+b);high.append(eigen[-1]+b)
    return np.array(low), np.array(high)


def fixtures():
    generator = np.random.default_rng(241124)
    checks = []
    for complex_key in (False, True):
        for repeated in (False, True):
            k = generator.normal(size=(2, 5, 3))
            q = generator.normal(size=(4, 3))
            if complex_key:
                k = k+1j*generator.normal(size=k.shape)
                q = q+1j*generator.normal(size=q.shape)
            if repeated:k[:] = k[:, :1]
            k /= np.linalg.norm(k, axis=-1, keepdims=True)
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            u = generator.normal(size=(2, 4, 5));b = generator.normal(size=4)
            mask = np.array([[True]*5, [True]*3+[False]*2])
            result = bounds(*(torch.from_numpy(a) for a in (k, u, b, mask, q)))
            error = 0.
            for i in range(2):
                nl, nh = numpy_reference(k[i, mask[i]], u[i][:, mask[i]], b)
                error = max(error, float(np.abs(result['query_low'][i].numpy()-nl).max()),
                            float(np.abs(result['query_high'][i].numpy()-nh).max()))
            assert error < 1e-7, error
            assert result['diagnostic']['maximum_original_query_outside_bounds'] < 1e-8
            assert result['diagnostic']['maximum_query_outside_simplex'] < 1e-8
            if repeated:
                target = (u*mask[:, None]).sum(-1)/mask.sum(-1)[:, None]+b
                assert np.max(np.abs(result['query_low'].numpy()-target)) < 1e-7
                assert np.max(np.abs(result['query_high'].numpy()-target)) < 1e-7
            checks.append(dict(complex=complex_key, repeated=repeated, max_error=error))
    return checks
