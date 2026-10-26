"""Key-set square-root measurement attention with an explicit null outcome.

The finite-dimensional Born probabilities admit an exact classical matrix
expression. This module makes no quantum speedup or task-utility claim.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable

MODES = ('disabled', 'quantum', 'fidelity', 'diagonal', 'key_mismatch')


class _InverseSquareRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        hermitian = (matrix + matrix.mH) / 2
        eigenvalues, vectors = torch.linalg.eigh(hermitian)
        if not torch.isfinite(eigenvalues).all() or (eigenvalues <= 0).any():
            raise ValueError('inverse square root requires a finite positive-definite matrix')
        roots = eigenvalues.sqrt()
        ctx.save_for_backward(roots, vectors)
        return (vectors / roots.unsqueeze(-2)) @ vectors.mH

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Exact divided difference of f(x)=x^(-1/2), including repeated roots.
        # Unlike differentiating eigenvectors, there is no eigenvalue-gap divisor.
        roots, vectors = ctx.saved_tensors
        left, right = roots.unsqueeze(-1), roots.unsqueeze(-2)
        loewner = -1 / (left * right * (left + right))
        grad = (grad_output + grad_output.mH) / 2
        return vectors @ (loewner * (vectors.mH @ grad @ vectors)) @ vectors.mH


def inverse_square_root(matrix):
    """Hermitian SPD inverse root; first derivatives only, in double precision."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('expected square matrices')
    dtype = torch.complex128 if matrix.is_complex() else torch.float64
    return _InverseSquareRoot.apply(matrix.to(dtype))


def _rotation(angle, axis):
    zero = torch.zeros_like(angle)
    if axis == 'y':
        c, s = (angle / 2).cos(), (angle / 2).sin()
        return torch.stack((c, -s, s, c), -1).reshape(*angle.shape, 2, 2).to(torch.complex128)
    a, b = torch.exp(-0.5j * angle), torch.exp(0.5j * angle)
    return torch.stack((a, zero, zero, b), -1).reshape(*angle.shape, 2, 2)


def _apply_gate(state, gate, wire):
    index = torch.arange(8, device=state.device)
    bit = 1 << (2 - wire)
    row = (index & bit) // bit
    return (gate[..., row, 0] * state[..., index & ~bit]
            + gate[..., row, 1] * state[..., index | bit])


class ThreeQubitMap(nn.Module):
    """8->6 input angles and two layers of local Rz/Ry/Rz plus fixed CZ ring."""
    def __init__(self, head_dim=8):
        super().__init__()
        if head_dim != 8:
            raise ValueError('Q-SRM v1 freezes head_dim=8')
        self.projection = nn.Linear(head_dim, 6)
        self.rotations = nn.Parameter(torch.empty(2, 3, 3).uniform_(-0.1, 0.1))
        basis = torch.arange(8)
        bits = [(basis >> (2 - w)) & 1 for w in range(3)]
        parity = (bits[0]*bits[1] + bits[1]*bits[2] + bits[2]*bits[0]) % 2
        self.register_buffer('cz_ring', (1 - 2*parity).to(torch.float64))

    def forward(self, features, *, capture=False):
        angles = math.pi * torch.tanh(self.projection(features)).to(torch.float64)
        state = torch.zeros((*features.shape[:-1], 8), device=features.device, dtype=torch.complex128)
        state[..., 0] = 1
        stages = [state] if capture else None
        for wire in range(3):
            state = _apply_gate(state, _rotation(angles[..., 2*wire], 'y'), wire)
            state = _apply_gate(state, _rotation(angles[..., 2*wire+1], 'z'), wire)
        if capture:
            stages.append(state)
        for layer in range(2):
            for wire in range(3):
                for index, axis in enumerate(('z', 'y', 'z')):
                    gate = _rotation(self.rotations[layer, wire, index].to(torch.float64), axis)
                    state = _apply_gate(state, gate, wire)
            state = state * self.cz_ring
            if capture:
                stages.append(state)
        return state, {'angles': angles, 'circuit_states': torch.stack(stages, -2)} if capture else {}


def _validate_states(query, key, mask):
    if query.ndim != 4 or query.shape != key.shape or query.shape[-1] != 8:
        raise ValueError('states must have equal [batch, head, token, 8] shape')
    if mask.shape != (query.shape[0], query.shape[2]) or mask.dtype != torch.bool:
        raise ValueError('mask must be boolean [batch, token]')
    if not mask.any(-1).all():
        raise ValueError('empty token support')
    if query.device != key.device or mask.device != query.device:
        raise ValueError('states and mask must share device')
    for state in (query, key):
        if not torch.isfinite(state).all():
            raise ValueError('nonfinite state')
        if not torch.allclose(state.abs().square().sum(-1), torch.ones_like(state.real[..., 0]), atol=1e-7, rtol=1e-7):
            raise ValueError('states must have unit norm')


def measure(query, key, mask, *, mode='quantum', ridge=0.01, capture=False):
    """Return token and null probabilities, never a token-only softmax.

Padded queries have zero token probability and null probability one. The
key_mismatch control cyclically moves valid key states while values stay put.
"""
    if mode not in MODES[1:] or ridge != 0.01:
        raise ValueError('unknown mode or changed frozen ridge')
    _validate_states(query, key, mask)
    query, key = query.to(torch.complex128), key.to(torch.complex128)
    b, heads, n, dim = query.shape
    permutation = torch.arange(n, device=key.device).expand(b, n).clone()
    if mode == 'key_mismatch':
        for row in range(b):
            valid = mask[row].nonzero(as_tuple=True)[0]
            permutation[row, valid] = valid.roll(1)
        key = key.gather(2, permutation[:, None, :, None].expand(b, heads, n, dim))
    support = mask[:, None, :, None, None]
    rho = key.unsqueeze(-1) * key.conj().unsqueeze(-2)
    if mode == 'diagonal':
        rho = torch.diag_embed(rho.diagonal(dim1=-2, dim2=-1))
    identity = torch.eye(dim, device=key.device, dtype=key.dtype)
    extra = {}
    if mode == 'fidelity':
        fidelity = (query.conj() @ key.transpose(-1, -2)).abs().square()
        fidelity = fidelity * mask[:, None, None, :]
        denominator = fidelity.sum(-1) + ridge
        attention = fidelity / denominator.unsqueeze(-1)
        null = ridge / denominator
        if capture:
            extra.update(fidelity=fidelity, denominator=denominator)
    else:
        gram = (rho * support).sum(-3) + ridge * identity
        inverse_root = inverse_square_root(gram)
        if mode == 'diagonal':
            diagonal_effects = (rho.diagonal(dim1=-2, dim2=-1).real * mask[:, None, :, None]
                                / gram.diagonal(dim1=-2, dim2=-1).real.unsqueeze(-2))
            attention = query.abs().square() @ diagonal_effects.transpose(-1, -2)
        else:
            whitened_keys = key @ inverse_root.transpose(-1, -2)
            attention = (query.conj() @ whitened_keys.transpose(-1, -2)).abs().square()
            attention = attention * mask[:, None, None, :]
        whitened_queries = query @ inverse_root.transpose(-1, -2)
        null = ridge * whitened_queries.abs().square().sum(-1)
        if capture:
            effects = inverse_root.unsqueeze(-3) @ (rho * support) @ inverse_root.unsqueeze(-3)
            null_effect = ridge * inverse_root @ inverse_root
            extra.update(gram=gram, inverse_root=inverse_root, spectrum=torch.linalg.eigvalsh(gram),
                         effects=effects, null_effect=null_effect,
                         completeness=effects.sum(-3)+null_effect-identity)
    query_support = mask[:, None, :]
    attention = attention * query_support.unsqueeze(-1)
    null = torch.where(query_support, null, torch.ones_like(null))
    if capture:
        extra.update(query_state=query, key_state=key, key_density=rho, permutation=permutation)
    return attention, null, extra


class SquareRootMeasurementAttention(nn.Module):
    def __init__(self, mode='quantum', head_dim=8):
        super().__init__()
        if mode not in MODES[1:]:
            raise ValueError('Q-SRM kernel requires an explicit active mode')
        self.mode = mode
        self.query_map = ThreeQubitMap(head_dim)
        self.key_map = ThreeQubitMap(head_dim)
        self.last = {}

    def forward(self, query, key, mask, *, capture=False):
        q, q_trace = self.query_map(query, capture=capture)
        k, k_trace = self.key_map(key, capture=capture)
        attention, null, trace = measure(q, k, mask, mode=self.mode, capture=capture)
        self.last = trace if capture else {}
        if capture:
            self.last.update({'query_'+name: value for name, value in q_trace.items()})
            self.last.update({'key_'+name: value for name, value in k_trace.items()})
            # Preserve the unpermuted producer output for mismatch lineage checks.
            self.last['encoded_key_state'] = k
        return attention, null
