"""Phase-sensitive quantum value gating for bounded attention experiments.

Q-PVG keeps the base attention distribution fixed by default and routes each
key's value through two learned value branches using a phase-sensitive
query-key overlap.  The optional ``score_value`` variant is deliberately
exposed as a separate ablation because it also changes attention scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.plugins.quantum_steering import _seeded_projection


VALUE_ROUTE_MODES = ("branch_interpolation", "scalar")
SCORE_MODES = ("value_only", "score_value")
READOUT_MODES = ("quantum", "classical")
PHASE_MODES = ("complex", "real_only", "random")


def _complex_state(
    features: torch.Tensor,
    rotation_projection: torch.Tensor,
    phase_projection: torch.Tensor,
    rotation_bias: torch.Tensor,
    phase_bias: torch.Tensor,
    *,
    phase_sign: float,
    angle_scale: float,
    register_qubits: int,
    depth: int,
    eps: float,
) -> torch.Tensor:
    """Prepare a data-dependent register state with explicit unitary gates.

    For every feature vector this constructs

    ``|psi(h)> = U_depth(h) ... U_1(h) |0...0>``

    where each layer contains data-dependent ``RY`` and ``RZ`` rotations and
    a nearest-neighbour CNOT ring.  The returned state is therefore a genuine
    statevector produced by a product of unitary operations, rather than a
    complex-valued softmax feature.  The overlap used by Q-PVG is subsequently
    ``<psi(q)|psi(k)> = <0|U_q(h_q)^dagger U_k(h_k)|0>``.

    ``rotation_projection`` and ``rotation_bias`` retain the historical
    buffer shapes; only their first ``register_qubits`` columns are used as
    per-qubit rotation angles.  This keeps old checkpoints loadable while
    making the state-preparation semantics explicit.
    """
    if features.shape[-1] != rotation_projection.shape[0]:
        raise ValueError("feature and rotation projection dimensions differ")
    state_dim = 2**register_qubits
    real_dtype = features.dtype if features.dtype in (torch.float32, torch.float64) else torch.float32
    initial_index = torch.zeros(features.shape[:-1], dtype=torch.long, device=features.device)
    initial_real = F.one_hot(initial_index, num_classes=state_dim).to(dtype=real_dtype)
    state = torch.complex(initial_real, torch.zeros_like(initial_real))

    rotation_angles = angle_scale * (
        features @ rotation_projection[..., :register_qubits]
        + rotation_bias[..., :register_qubits]
    )
    phase_angles = phase_sign * angle_scale * (
        features @ phase_projection[..., :register_qubits]
        + phase_bias[..., :register_qubits]
    )
    for _ in range(depth):
        for qubit in range(register_qubits):
            state = _apply_ry(state, rotation_angles, qubit)
            state = _apply_rz(state, phase_angles, qubit)
        if register_qubits > 1:
            for control in range(register_qubits):
                state = _apply_cnot(state, control, (control + 1) % register_qubits)

    norm = state.abs().square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return state / norm


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a differentiable single-qubit RY gate to a batched statevector."""
    register_qubits = int(math.log2(state.shape[-1]))
    left = 2**qubit
    right = 2 ** (register_qubits - qubit - 1)
    view = state.reshape(*state.shape[:-1], left, 2, right)
    theta = angles[..., qubit].unsqueeze(-1).unsqueeze(-1)
    cosine = torch.cos(theta / 2.0)
    sine = torch.sin(theta / 2.0)
    zero = cosine * view[..., 0, :] - sine * view[..., 1, :]
    one = sine * view[..., 0, :] + cosine * view[..., 1, :]
    return torch.stack((zero, one), dim=-2).reshape_as(state)


def _apply_rz(state: torch.Tensor, angles: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a differentiable single-qubit RZ gate to a batched statevector."""
    register_qubits = int(math.log2(state.shape[-1]))
    left = 2**qubit
    right = 2 ** (register_qubits - qubit - 1)
    view = state.reshape(*state.shape[:-1], left, 2, right)
    theta = angles[..., qubit].unsqueeze(-1).unsqueeze(-1)
    half = theta / 2.0
    phase_zero = torch.complex(torch.cos(half), -torch.sin(half))
    phase_one = torch.complex(torch.cos(half), torch.sin(half))
    zero = phase_zero * view[..., 0, :]
    one = phase_one * view[..., 1, :]
    return torch.stack((zero, one), dim=-2).reshape_as(state)


def _apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply a CNOT by permuting computational-basis amplitudes."""
    register_qubits = int(math.log2(state.shape[-1]))
    if control == target:
        raise ValueError("CNOT control and target must differ")
    indices = torch.arange(state.shape[-1], device=state.device)
    # ``_apply_ry/_apply_rz`` expose qubit 0 as the leading basis axis, so
    # convert that logical numbering to the integer basis-bit convention.
    control_bit = (indices >> (register_qubits - 1 - control)) & 1
    target_bit = register_qubits - 1 - target
    permutation = indices ^ (control_bit << target_bit)
    return state.index_select(-1, permutation)


def _masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(masked, dim=-1) * mask.to(dtype=scores.dtype)
    denominator = attention.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attention / denominator


@dataclass(frozen=True)
class QPVGConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    register_qubits: int = 2
    depth: int = 2
    angle_scale: float = 1.0
    gate_temperature: float = 1.0
    score_gain: float = 0.25
    value_route_mode: str = "branch_interpolation"
    score_mode: str = "value_only"
    readout_mode: str = "quantum"
    phase_mode: str = "complex"
    seed: int = 211
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.register_qubits <= 0 or self.depth <= 0:
            raise ValueError("register_qubits and depth must be positive")
        if self.angle_scale <= 0.0 or self.gate_temperature <= 0.0:
            raise ValueError("angle_scale and gate_temperature must be positive")
        if self.score_gain < 0.0:
            raise ValueError("score_gain must be non-negative")
        if self.value_route_mode not in VALUE_ROUTE_MODES:
            raise ValueError(f"value_route_mode must be one of {VALUE_ROUTE_MODES}")
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}")
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}")
        if self.phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class PhaseValueGatingKernel(nn.Module):
    """Phase-sensitive query-key gating followed by value routing."""

    mechanism_name = "q_pvg"

    def __init__(self, config: QPVGConfig) -> None:
        super().__init__()
        self.config = config
        self.state_dim = 2 ** config.register_qubits

        self.register_buffer(
            "query_amplitude_projections",
            torch.stack(
                [
                    _seeded_projection(
                        config.head_dim,
                        self.state_dim,
                        config.seed + 17 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        self.register_buffer(
            "key_amplitude_projections",
            torch.stack(
                [
                    _seeded_projection(
                        config.head_dim,
                        self.state_dim,
                        config.seed + 1013 + 19 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        self.register_buffer(
            "query_phase_projections",
            torch.stack(
                [
                    _seeded_projection(
                        config.head_dim,
                        self.state_dim,
                        config.seed + 2039 + 23 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        self.register_buffer(
            "key_phase_projections",
            torch.stack(
                [
                    _seeded_projection(
                        config.head_dim,
                        self.state_dim,
                        config.seed + 3079 + 29 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 4099)
        shape = (config.num_layers, config.num_heads, self.state_dim)
        self.query_amplitude_bias = nn.Parameter(
            torch.empty(shape).uniform_(-0.2, 0.2, generator=generator)
        )
        self.key_amplitude_bias = nn.Parameter(
            torch.empty(shape).uniform_(-0.2, 0.2, generator=generator)
        )
        self.query_phase_bias = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi, math.pi, generator=generator)
        )
        self.key_phase_bias = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi, math.pi, generator=generator)
        )
        self.gate_real_weight = nn.Parameter(torch.ones(config.num_layers, config.num_heads))
        self.gate_imag_weight = nn.Parameter(torch.full((config.num_layers, config.num_heads), 0.75))
        self.value_branch0 = nn.Parameter(
            torch.eye(config.head_dim).repeat(config.num_layers, config.num_heads, 1, 1)
        )
        branch1 = torch.eye(config.head_dim).repeat(config.num_layers, config.num_heads, 1, 1)
        branch1 = branch1 + 0.05 * torch.randn(branch1.shape, generator=generator)
        self.value_branch1 = nn.Parameter(branch1)
        self.scalar_value_logit = nn.Parameter(torch.zeros(config.num_layers, config.num_heads))

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.mechanism_name,
            "config": asdict(self.config),
            "mechanism": {
                "state_preparation": (
                    "data-dependent RY/RZ rotations followed by a nearest-neighbour "
                    "CNOT ring from |0...0>"
                ),
                "alignment": "z=<0|U_q(h_q)^dagger U_k(h_k)|0>, read as Re(z), Im(z)",
                "gate": "sigmoid((a Re(z) + b Im(z)) / temperature)",
                "value_route": "g R1(v) + (1-g) R0(v) or g v",
                "default_score_policy": "base attention fixed; value route only",
                "score_value_ablation": "gate-derived centered score adjustment",
                "classical_control": (
                    "same state preparation and parameter shapes; classical readout "
                    "uses only Re(z) and disables the Im(z) gate channel"
                ),
            },
        }

    def _states(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        phase_sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_states = []
        key_states = []
        for head in range(self.config.num_heads):
            q_state = _complex_state(
                query[:, head],
                self.query_amplitude_projections[head],
                self.query_phase_projections[head],
                self.query_amplitude_bias[layer_index, head],
                self.query_phase_bias[layer_index, head],
                phase_sign=1.0,
                angle_scale=self.config.angle_scale,
                register_qubits=self.config.register_qubits,
                depth=self.config.depth,
                eps=self.config.eps,
            )
            k_state = _complex_state(
                key[:, head],
                self.key_amplitude_projections[head],
                self.key_phase_projections[head],
                self.key_amplitude_bias[layer_index, head],
                self.key_phase_bias[layer_index, head],
                phase_sign=phase_sign,
                angle_scale=self.config.angle_scale,
                register_qubits=self.config.register_qubits,
                depth=self.config.depth,
                eps=self.config.eps,
            )
            query_states.append(q_state)
            key_states.append(k_state)
        return torch.stack(query_states, dim=1), torch.stack(key_states, dim=1)

    def _alignment(self, query_state: torch.Tensor, key_state: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhqd,bhkd->bhqk", query_state.conj(), key_state)

    def _route_values(
        self,
        value: torch.Tensor,
        gate: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value0 = torch.einsum(
            "bhkd,hdm->bhkm", value, self.value_branch0[layer_index]
        )
        value1 = torch.einsum(
            "bhkd,hdm->bhkm", value, self.value_branch1[layer_index]
        )
        if self.config.value_route_mode == "branch_interpolation":
            routed = gate.unsqueeze(-1) * value1.unsqueeze(2) + (
                1.0 - gate.unsqueeze(-1)
            ) * value0.unsqueeze(2)
        else:
            scalar = 2.0 * torch.sigmoid(self.scalar_value_logit[layer_index])
            routed = gate.unsqueeze(-1) * value.unsqueeze(2) * scalar.view(1, -1, 1, 1, 1)
        return value0, value1, routed

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        scores: torch.Tensor | None = None,
        layer_index: int,
        attention_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        phase_sign: float = 1.0,
        return_trace: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if value is None or scores is None:
            raise ValueError("Q-PVG requires value and pre-softmax scores")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or scores.ndim != 4:
            raise ValueError("query, key, value, and scores must be rank-4 tensors")
        if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
            raise ValueError("query/key/value dimensions are incompatible")
        if layer_index < 0 or layer_index >= self.config.num_layers:
            raise ValueError("layer_index is outside configured layers")
        if attention_mask.shape != (scores.shape[0], key.shape[2]):
            raise ValueError("attention_mask must match key tokens")
        if query_mask is None:
            query_mask = torch.ones(
                query.shape[0], query.shape[2], dtype=torch.bool, device=query.device
            )
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        query_mask_4d = query_mask[:, None, :, None].to(dtype=scores.dtype)
        base_attention = _masked_softmax(scores, key_mask, self.config.eps)
        query_state, key_state = self._states(
            query, key, layer_index=layer_index, phase_sign=phase_sign
        )
        alignment = self._alignment(query_state, key_state)
        real_part = alignment.real
        imag_part = alignment.imag
        if self.config.readout_mode == "classical" or self.config.phase_mode == "real_only":
            gate_imag = torch.zeros_like(imag_part)
        elif self.config.phase_mode == "random":
            random_index = torch.arange(
                imag_part.numel(), device=imag_part.device, dtype=imag_part.dtype
            ).reshape(imag_part.shape)
            gate_imag = torch.sin(random_index * 12.9898 + float(self.config.seed))
        else:
            gate_imag = imag_part
        logits = (
            self.gate_real_weight[layer_index].view(1, -1, 1, 1) * real_part
            + self.gate_imag_weight[layer_index].view(1, -1, 1, 1) * gate_imag
        ) / self.config.gate_temperature
        gate = torch.sigmoid(logits)
        gate = gate * key_mask.to(dtype=gate.dtype) * query_mask_4d
        value0, value1, routed_values = self._route_values(
            value, gate, layer_index=layer_index
        )
        score_adjustment = torch.zeros_like(scores)
        attention = base_attention
        if self.config.score_mode == "score_value":
            centered_gate = gate - (
                gate * key_mask.to(dtype=gate.dtype)
            ).sum(dim=-1, keepdim=True) / key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            score_adjustment = self.config.score_gain * centered_gate * key_mask.to(dtype=scores.dtype)
            attention = _masked_softmax(scores + score_adjustment, key_mask, self.config.eps)
        output = torch.einsum("bhqk,bhqkd->bhqd", attention, routed_values)
        output = output * query_mask_4d
        if not return_trace:
            return output
        trace = {
            "query_state_real": query_state.real,
            "query_state_imag": query_state.imag,
            "key_state_real": key_state.real,
            "key_state_imag": key_state.imag,
            "query_state_norm": query_state.abs().square().sum(dim=-1).sqrt(),
            "key_state_norm": key_state.abs().square().sum(dim=-1).sqrt(),
            "alignment_real": real_part,
            "alignment_imag": imag_part,
            "gate_imag": gate_imag,
            "gate": gate,
            "base_attention": base_attention,
            "score_adjustment": score_adjustment,
            "attention": attention,
            "value_branch0": value0,
            "value_branch1": value1,
            "routed_values": routed_values,
            "output": output,
        }
        return output, trace


def build_q_pvg(config: QPVGConfig) -> PhaseValueGatingKernel:
    """Construct a Q-PVG kernel from a validated configuration."""
    return PhaseValueGatingKernel(config)


__all__ = [
    "READOUT_MODES",
    "PHASE_MODES",
    "SCORE_MODES",
    "VALUE_ROUTE_MODES",
    "PhaseValueGatingKernel",
    "QPVGConfig",
    "build_q_pvg",
]
