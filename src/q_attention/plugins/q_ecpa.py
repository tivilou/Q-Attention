"""Entity-conditioned complex path propagation, with explicit matched controls.

H[destination, source]; attention P[query/source, key/destination]. No targets
are accepted. The graph has adjacent and skip-one edges (triangles).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn


MODES = ("disabled", "quantum", "dephased", "classical", "random_phase")


@dataclass(frozen=True)
class ECPAConfig:
    heads: int = 2
    head_dim: int = 8
    steps: int = 3
    delta: float = 0.7
    phase_max: float = math.pi
    alpha: float = 0.1
    clip: float = 4.0
    eps: float = 1e-6
    random_seed: int = 1729

    def __post_init__(self):
        if self.heads < 1 or self.head_dim < 2 or self.steps < 1:
            raise ValueError("invalid ECPA dimensions/steps")
        if not (0 < self.delta <= 2 and 0 < self.phase_max <= math.pi):
            raise ValueError("invalid propagation time/phase bound")
        if not (0 < self.alpha <= 0.5 and self.clip > 0 and 0 < self.eps < 1):
            raise ValueError("invalid residual bound")


def propagate(hamiltonian: torch.Tensor, steps: int, delta: float):
    """Preserve all steps for optional selected-sample capture only."""
    unitary = torch.matrix_exp(-1j * delta * hamiltonian)
    amplitude = torch.eye(unitary.shape[-1], device=unitary.device,
                          dtype=unitary.dtype).expand_as(unitary)
    single = unitary.abs().square()
    dephased = torch.eye(unitary.shape[-1], device=unitary.device,
                         dtype=single.dtype).expand_as(single)
    amplitudes, probabilities = [], []
    for _ in range(steps):
        amplitude = unitary @ amplitude
        dephased = single @ dephased
        amplitudes.append(amplitude)
        probabilities.append(dephased)
    return (unitary, torch.stack(amplitudes, -3), single,
            torch.stack(probabilities, -3))


class EntityCoherentPath(nn.Module):
    def __init__(self, config: ECPAConfig, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unsupported ECPA mode: {mode}")
        self.config, self.mode = config, mode
        d, h = config.head_dim, config.heads
        # Same trainable parameter budget for Q / D / classical.
        self.condition = nn.Linear(2 * d, d * d)
        nn.init.normal_(self.condition.weight, std=0.12)
        nn.init.normal_(self.condition.bias, std=0.12)
        generator = torch.Generator().manual_seed(config.random_seed)
        self.register_buffer("random_operator", torch.randn(h, d, d, generator=generator))
        if mode in ("disabled", "random_phase"):
            self.condition.requires_grad_(False)

    def forward(self, query, scores, mask, subject, obj, *, capture=False):
        c = self.config
        if query.ndim != 4 or query.shape[1] != c.heads or query.shape[-1] != c.head_dim:
            raise ValueError("query shape must be B,H,N,D matching ECPA config")
        b, h, n, d = query.shape
        if scores.shape != (b, h, n, n):
            raise ValueError("score shape mismatch")
        mask = mask.to(device=query.device, dtype=torch.bool)
        if mask.shape != (b, n) or subject.shape != (b, n) or obj.shape != (b, n):
            raise ValueError("mask shape mismatch")
        if (~mask.any(-1)).any() or ((subject.bool() | obj.bool()) & ~mask).any():
            raise ValueError("empty input or entity outside valid tokens")
        if (subject.sum(-1) == 0).any() or (obj.sum(-1) == 0).any():
            raise ValueError("entity spans must be nonempty")
        if self.mode == "disabled":
            return scores, {}
        z = torch.nn.functional.normalize(query, dim=-1, eps=c.eps)
        def pool(span):
            weight = span.to(query)[:, None, :, None]
            return (z * weight).sum(-2) / weight.sum(-2)
        pair = torch.cat((pool(subject), pool(obj)), -1)
        operator = self.condition(pair).reshape(b, h, d, d)
        if self.mode == "random_phase":
            operator = self.random_operator[None].expand(b, -1, -1, -1)
        skew = 0.5 * (operator - operator.transpose(-1, -2))
        phase = c.phase_max * torch.tanh(torch.einsum("bhid,bhde,bhje->bhij", z, skew, z))
        # Explicit antisymmetrization also removes floating-point diagonal drift.
        phase = 0.5 * (phase - phase.transpose(-1, -2))
        position = torch.arange(n, device=query.device)
        distance = (position[:, None] - position[None, :]).abs()
        edges = (distance > 0) & (distance <= 2)
        support = mask[:, None, :, None] & mask[:, None, None, :]
        weights = (edges[None, None] & support).to(query)
        degree = weights.sum(-1)
        denom = (degree[..., :, None] * degree[..., None, :]).clamp_min(1).sqrt()
        graph = weights / denom
        hamiltonian = torch.polar(graph.expand(b, h, n, n), phase)
        unitary, amplitudes, single, dephased = propagate(hamiltonian, c.steps, c.delta)
        pq = amplitudes[..., -1, :, :].abs().square().transpose(-1, -2)
        pd = dephased[..., -1, :, :].transpose(-1, -2)
        # Strong classical comparator: same phase features become real edge
        # logits. A self edge permits staying in place, like quantum propagation.
        self_edges = torch.eye(n, device=query.device, dtype=torch.bool)
        classical_support = (edges | self_edges)[None, None] & support
        classical_scores = phase.transpose(-1, -2).masked_fill(~classical_support, -1e4)
        transition = torch.softmax(classical_scores, -1) * mask[:, None, :, None]
        classical = torch.eye(n, device=query.device, dtype=query.dtype).expand(b, h, n, n)
        classical_steps = []
        for _ in range(c.steps):
            classical = classical @ transition
            classical_steps.append(classical)
        classical_steps = torch.stack(classical_steps, -3)
        probability = {"quantum": pq, "random_phase": pq, "dephased": pd,
                       "classical": classical}[self.mode]
        valid = support.expand(b, h, n, n)
        logged = torch.log(probability + c.eps)
        reference = (logged * valid).sum(-1, keepdim=True) / mask.sum(-1)[:, None, None, None]
        residual = c.alpha * (logged - reference).clamp(-c.clip, c.clip) * valid
        adjusted = scores + residual
        state = {}
        if capture:
            state = dict(entity_pair=pair, skew=skew, edges=weights.expand(b,h,n,n),
                         degree=degree.expand(b,h,n), phase=phase, hamiltonian=hamiltonian,
                         unitary=unitary, amplitudes=amplitudes, single=single,
                         dephased_steps=dephased, pq=pq, pd=pd, transition=transition,
                         classical_steps=classical_steps, probability=probability,
                         residual=residual)
        return adjusted, state
