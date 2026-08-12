"""Counterfactual token-evidence selectors for attention-score kernels."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_steering import (
    _apply_cnot,
    _apply_ry,
    _data_reuploading_state,
    _raw_gain,
    _seeded_projection,
)


EVIDENCE_SELECTOR_TYPES = (
    "quantum",
    "qness",
    "qness_commuting",
    "qness_separable",
    "qness_phase_scrambled",
    "qness_dephased",
    "qness_classical",
    "classical",
    "classical_strong",
)
EVIDENCE_VIEW_CHOICES = ("full", "keep", "drop", "random_keep", "random_drop")
EVIDENCE_READOUT_CHOICES = (
    "joint_observable",
    "factorized_observable",
    "connected_relation_token",
)
EVIDENCE_CORRELATION_MODES = (
    "connected",
    "total",
    "multiscale",
    "correlation_gated",
    "signed_gated",
    "standardized_connected",
    "standardized_signed_gated",
    "phase_selective",
    "phase_rotated",
    "dual_channel",
    "born_reliability",
)
EVIDENCE_CORRELATION_CHANNELS = (
    "pre_entanglement_product",
    "post_entanglement_product",
    "connected",
    "total",
    "multiscale",
    "correlation_gated",
    "signed_gated",
    "standardized_connected",
    "standardized_signed_gated",
    "phase_selective",
    "phase_rotated",
    "dual_channel",
    "born_reliability",
)
EVIDENCE_WEIGHT_MODES = ("positive_simplex", "signed_centered_l1")
EVIDENCE_INTERVENTION_MODES = ("kernel_scale", "direct_bias")
EVIDENCE_DIRECT_BIAS_MODES = (
    "centered",
    "positive_excess",
    "evidence_weighted_centered",
)
EVIDENCE_GATE_CALIBRATION_MODES = ("none", "context_budget")
EVIDENCE_VIEW_SCORE_MODES = ("positive", "polarity_magnitude")
EVIDENCE_TASK_READOUT_MODES = ("shared", "dual")
EVIDENCE_MEASUREMENT_MODES = (
    "fixed",
    "relation_conditioned",
    "relation_frame",
    "relation_frame_bank",
    "relation_frame_coherent",
    "relation_directional",
    "entanglement_directional",
    "entanglement_phase_offset",
)
EVIDENCE_MEASUREMENT_FRAME_VIEWS = ("full", "z", "x")
RELATION_EVIDENCE_ANCHOR_MODES = ("entity_pair", "leave_one_out_context")
QNESS_CONTROL_MODES = (
    "none",
    "commuting",
    "separable",
    "phase_scrambled",
    "dephased",
)


@dataclass(frozen=True)
class RelationEvidenceSelectorConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_qubits: int = 4
    depth: int = 2
    angle_scale: float = 1.0
    mask_floor: float = 0.05
    evidence_gate_calibration: str = "none"
    evidence_budget: float = 0.35
    evidence_view_score_mode: str = "positive"
    evidence_task_readout: str = "shared"
    initial_sharpness: float = 2.0
    evidence_readout: str = "joint_observable"
    evidence_correlation_mode: str = "connected"
    relation_anchor_mode: str = "entity_pair"
    evidence_weight_mode: str = "positive_simplex"
    evidence_measurement_mode: str = "fixed"
    max_conditioning_gain: float = 2.0
    initial_conditioning_gain: float = 0.5
    relation_frame_scale: float = 1.0
    max_frame_fusion_gain: float = 2.0
    initial_frame_fusion_gain: float = 1.0
    intervention_mode: str = "kernel_scale"
    direct_bias_mode: str = "centered"
    max_direct_gain: float = 1.0
    initial_direct_gain: float = 0.1
    cross_entanglement: bool = True
    qness_control: str = "none"
    quantum_diagnostic_limit: int = 64
    seed: int = 71
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")
        if self.num_qubits <= 0 or self.depth <= 0:
            raise ValueError("num_qubits and depth must be positive")
        if self.angle_scale <= 0 or self.initial_sharpness <= 0:
            raise ValueError("angle_scale and initial_sharpness must be positive")
        if not 0.0 < self.mask_floor < 1.0:
            raise ValueError("mask_floor must lie inside (0, 1)")
        if self.evidence_gate_calibration not in EVIDENCE_GATE_CALIBRATION_MODES:
            raise ValueError(
                "evidence_gate_calibration must be one of "
                f"{EVIDENCE_GATE_CALIBRATION_MODES}"
            )
        if not 0.0 < self.evidence_budget < 1.0:
            raise ValueError("evidence_budget must lie inside (0, 1)")
        if self.evidence_view_score_mode not in EVIDENCE_VIEW_SCORE_MODES:
            raise ValueError(
                "evidence_view_score_mode must be one of "
                f"{EVIDENCE_VIEW_SCORE_MODES}"
            )
        if self.evidence_task_readout not in EVIDENCE_TASK_READOUT_MODES:
            raise ValueError(
                "evidence_task_readout must be one of "
                f"{EVIDENCE_TASK_READOUT_MODES}"
            )
        if (
            self.evidence_task_readout == "dual"
            and self.evidence_readout != "connected_relation_token"
        ):
            raise ValueError(
                "dual task readout requires connected_relation_token"
            )
        if self.evidence_readout not in EVIDENCE_READOUT_CHOICES:
            raise ValueError(
                f"evidence_readout must be one of {EVIDENCE_READOUT_CHOICES}"
            )
        if self.evidence_correlation_mode not in EVIDENCE_CORRELATION_MODES:
            raise ValueError(
                "evidence_correlation_mode must be one of "
                f"{EVIDENCE_CORRELATION_MODES}"
            )
        if (
            self.evidence_correlation_mode != "connected"
            and self.evidence_readout != "connected_relation_token"
        ):
            raise ValueError(
                "non-default evidence_correlation_mode requires "
                "connected_relation_token"
            )
        if self.evidence_readout == "connected_relation_token":
            if self.num_qubits < 4 or self.num_qubits % 2:
                raise ValueError(
                    "connected_relation_token requires an even num_qubits >= 4"
                )
        if (
            self.evidence_correlation_mode
            in {
                "dual_channel",
                "born_reliability",
                "phase_selective",
                "phase_rotated",
            }
            and self.evidence_measurement_mode
            not in {
                "relation_frame_bank",
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }
        ):
            raise ValueError(
                f"{self.evidence_correlation_mode} requires a two-frame "
                "relation measurement mode"
            )
        if self.relation_anchor_mode not in RELATION_EVIDENCE_ANCHOR_MODES:
            raise ValueError(
                "relation_anchor_mode must be one of "
                f"{RELATION_EVIDENCE_ANCHOR_MODES}"
            )
        if self.evidence_weight_mode not in EVIDENCE_WEIGHT_MODES:
            raise ValueError(
                f"evidence_weight_mode must be one of {EVIDENCE_WEIGHT_MODES}"
            )
        if (
            self.evidence_weight_mode == "signed_centered_l1"
            and self.evidence_readout != "connected_relation_token"
        ):
            raise ValueError(
                "signed_centered_l1 is only defined for connected_relation_token"
            )
        if self.evidence_measurement_mode not in EVIDENCE_MEASUREMENT_MODES:
            raise ValueError(
                "evidence_measurement_mode must be one of "
                f"{EVIDENCE_MEASUREMENT_MODES}"
            )
        if self.evidence_measurement_mode in {
            "relation_conditioned",
            "relation_frame",
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            if self.evidence_readout != "connected_relation_token":
                raise ValueError(
                    "relation-aware measurement requires "
                    "connected_relation_token"
                )
            if self.evidence_weight_mode != "signed_centered_l1":
                raise ValueError(
                    "relation-aware measurement requires signed_centered_l1"
                )
            if self.num_qubits != 4:
                raise ValueError(
                    "relation-aware measurement currently requires 4 qubits"
                )
        if (
            self.evidence_measurement_mode
            in {"entanglement_directional", "entanglement_phase_offset"}
            and self.evidence_correlation_mode != "phase_selective"
        ):
            raise ValueError(
                "entanglement-derived measurement requires "
                "evidence_correlation_mode='phase_selective'"
            )
        if self.max_conditioning_gain <= 0.0:
            raise ValueError("max_conditioning_gain must be positive")
        if not (
            -self.max_conditioning_gain
            < self.initial_conditioning_gain
            < self.max_conditioning_gain
        ):
            raise ValueError(
                "initial_conditioning_gain must lie inside "
                "(-max_conditioning_gain, max_conditioning_gain)"
            )
        if self.relation_frame_scale <= 0.0:
            raise ValueError("relation_frame_scale must be positive")
        if self.max_frame_fusion_gain <= 0.0:
            raise ValueError("max_frame_fusion_gain must be positive")
        if not (
            -self.max_frame_fusion_gain
            < self.initial_frame_fusion_gain
            < self.max_frame_fusion_gain
        ):
            raise ValueError(
                "initial_frame_fusion_gain must lie inside "
                "(-max_frame_fusion_gain, max_frame_fusion_gain)"
            )
        if self.intervention_mode not in EVIDENCE_INTERVENTION_MODES:
            raise ValueError(
                f"intervention_mode must be one of {EVIDENCE_INTERVENTION_MODES}"
            )
        if self.direct_bias_mode not in EVIDENCE_DIRECT_BIAS_MODES:
            raise ValueError(
                f"direct_bias_mode must be one of {EVIDENCE_DIRECT_BIAS_MODES}"
            )
        if self.max_direct_gain <= 0.0:
            raise ValueError("max_direct_gain must be positive")
        if not -self.max_direct_gain < self.initial_direct_gain < self.max_direct_gain:
            raise ValueError(
                "initial_direct_gain must lie inside (-max_direct_gain, max_direct_gain)"
            )
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.qness_control not in QNESS_CONTROL_MODES:
            raise ValueError(
                f"qness_control must be one of {QNESS_CONTROL_MODES}"
            )
        if self.quantum_diagnostic_limit < 0:
            raise ValueError("quantum_diagnostic_limit must be non-negative")


def _masked_head_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weights = mask[:, None, :, None].to(device=values.device, dtype=values.dtype)
    return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(eps)


def _join_register_states(
    relation_states: torch.Tensor,
    token_states: torch.Tensor,
) -> torch.Tensor:
    if relation_states.ndim != 2 or token_states.ndim != 2:
        raise ValueError("register states must be matrices")
    if relation_states.shape[0] != token_states.shape[0]:
        raise ValueError("relation and token registers must share the batch size")
    return torch.einsum("bi,bj->bij", relation_states, token_states).reshape(
        relation_states.shape[0], -1
    )


def _cross_register_entangle(
    state: torch.Tensor,
    register_qubits: int,
) -> torch.Tensor:
    total_qubits = 2 * register_qubits
    for qubit in range(register_qubits):
        state = _apply_cnot(state, qubit, register_qubits + qubit, total_qubits)
    if register_qubits > 1:
        state = _apply_cnot(state, register_qubits, 1, total_qubits)
    return state


def _z_signs(
    basis: torch.Tensor,
    qubits: tuple[int, ...],
    total_qubits: int,
) -> torch.Tensor:
    parity = torch.zeros_like(basis)
    for qubit in qubits:
        bit = (basis >> (total_qubits - qubit - 1)).bitwise_and(1)
        parity = parity.bitwise_xor(bit)
    return 1.0 - 2.0 * parity.to(dtype=torch.float32)


def _relation_token_z_correlations(
    state: torch.Tensor,
    register_qubits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint, marginal_product, connected, _relation, _token = (
        _relation_token_z_statistics(state, register_qubits)
    )
    return joint, marginal_product, connected


def _relation_token_z_statistics(
    state: torch.Tensor,
    register_qubits: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    total_qubits = 2 * register_qubits
    basis = torch.arange(state.shape[-1], device=state.device)
    probabilities = state.square()
    relation_expectations = []
    token_expectations = []
    for qubit in range(register_qubits):
        relation_signs = _z_signs(basis, (qubit,), total_qubits).to(state.dtype)
        token_signs = _z_signs(
            basis,
            (register_qubits + qubit,),
            total_qubits,
        ).to(state.dtype)
        relation_expectations.append((probabilities * relation_signs).sum(dim=-1))
        token_expectations.append((probabilities * token_signs).sum(dim=-1))
    relation_expectations = torch.stack(relation_expectations, dim=-1)
    token_expectations = torch.stack(token_expectations, dim=-1)
    joint = []
    for relation_qubit in range(register_qubits):
        for token_qubit in range(register_qubits):
            signs = _z_signs(
                basis,
                (relation_qubit, register_qubits + token_qubit),
                total_qubits,
            ).to(state.dtype)
            joint.append((probabilities * signs).sum(dim=-1))
    joint_tensor = torch.stack(joint, dim=-1)
    marginal_product = (
        relation_expectations.unsqueeze(-1) * token_expectations.unsqueeze(-2)
    ).reshape(state.shape[0], -1)
    connected = joint_tensor - marginal_product
    return (
        joint_tensor,
        marginal_product,
        connected,
        relation_expectations,
        token_expectations,
    )


def _x_expectation(
    state: torch.Tensor,
    qubit: int,
    total_qubits: int,
) -> torch.Tensor:
    basis = torch.arange(state.shape[-1], device=state.device)
    bit = 1 << (total_qubits - qubit - 1)
    return (state * state[:, basis.bitwise_xor(bit)]).sum(dim=-1)


def _relation_x_token_z_correlations(
    state: torch.Tensor,
    register_qubits: int,
) -> torch.Tensor:
    """Measure X_relation Z_token observables on a real joint state."""
    total_qubits = 2 * register_qubits
    basis = torch.arange(state.shape[-1], device=state.device)
    expectations: list[torch.Tensor] = []
    for relation_qubit in range(register_qubits):
        relation_bit = 1 << (total_qubits - relation_qubit - 1)
        flipped = basis.bitwise_xor(relation_bit)
        for token_qubit in range(register_qubits):
            token_sign = _z_signs(
                basis,
                (register_qubits + token_qubit,),
                total_qubits,
            ).to(device=state.device, dtype=state.dtype)
            expectations.append(
                (state * state[:, flipped] * token_sign).sum(dim=-1)
            )
    return torch.stack(expectations, dim=-1)


def _pure_state_off_diagonal_norm(state: torch.Tensor) -> torch.Tensor:
    probabilities = state.square()
    return torch.sqrt(
        (1.0 - probabilities.square().sum(dim=-1)).clamp_min(0.0)
    )


def _pure_state_register_mutual_information(
    state: torch.Tensor,
    register_qubits: int,
    eps: float,
) -> torch.Tensor:
    register_dim = 2**register_qubits
    amplitudes = state.reshape(state.shape[0], register_dim, register_dim)
    reduced = torch.matmul(amplitudes, amplitudes.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(reduced).clamp_min(0.0)
    normalized = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(
        normalized
        * torch.where(
            normalized > eps,
            normalized.clamp_min(eps).log(),
            torch.zeros_like(normalized),
        )
    ).sum(dim=-1)
    return 2.0 * entropy


def _connected_z_correlations(
    state: torch.Tensor,
    register_qubits: int,
) -> torch.Tensor:
    return _relation_token_z_correlations(state, register_qubits)[2]


def _total_z_correlations(
    state: torch.Tensor,
    register_qubits: int,
) -> torch.Tensor:
    _joint, marginal_product, connected = _relation_token_z_correlations(
        state,
        register_qubits,
    )
    return marginal_product + connected


def _interleave_frame_banks(
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> torch.Tensor:
    if primary.shape != secondary.shape or primary.shape[-1] % 2:
        raise ValueError("dual-channel frame banks must have matching even shapes")
    primary_frames = primary.reshape(primary.shape[0], 2, -1)
    secondary_frames = secondary.reshape(secondary.shape[0], 2, -1)
    return torch.cat((primary_frames, secondary_frames), dim=-1).reshape(
        primary.shape[0], -1
    )


def _classical_dual_channel_features(
    features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nonlinear_features = features * features.abs()
    zeros = torch.zeros_like(features)
    primary = _interleave_frame_banks(features, zeros)
    secondary = _interleave_frame_banks(zeros, nonlinear_features)
    dual = primary + secondary
    channels = {
        "pre_entanglement_product": secondary,
        "post_entanglement_product": secondary,
        "connected": primary,
        "total": dual,
        "multiscale": dual,
        "correlation_gated": primary,
        "signed_gated": primary,
        "standardized_connected": primary,
        "standardized_signed_gated": primary,
        "phase_selective": primary,
        "phase_rotated": primary,
        "dual_channel": dual,
    }
    return dual, channels


def _two_qubit_pauli_features(state: torch.Tensor) -> torch.Tensor:
    """Measure Z0, Z1, Z0Z1, and X0X1 on a real two-qubit state."""
    if state.ndim != 2 or state.shape[-1] != 4:
        raise ValueError("two-qubit states must have shape (batch, 4)")
    probabilities = state.square()
    z0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
    z1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
    zz = probabilities[:, 0] - probabilities[:, 1] - probabilities[:, 2] + probabilities[:, 3]
    xx = (state * state[:, (3, 2, 1, 0)]).sum(dim=-1)
    return torch.stack((z0, z1, zz, xx), dim=-1)


def _local_bloch_angles(
    z_expectation: torch.Tensor,
    x_expectation: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if z_expectation.shape != x_expectation.shape:
        raise ValueError("local Z and X expectations must have matching shapes")
    active = z_expectation.square() + x_expectation.square() > eps**2
    safe_z = torch.where(active, z_expectation, torch.ones_like(z_expectation))
    safe_x = torch.where(active, x_expectation, torch.zeros_like(x_expectation))
    return torch.atan2(safe_x, safe_z)


def _two_qubit_local_bloch_angles(
    state: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the real-state Bloch angle of each local qubit."""
    if state.ndim != 2 or state.shape[-1] != 4:
        raise ValueError("two-qubit states must have shape (batch, 4)")
    probabilities = state.square()
    z0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
    z1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
    x0 = 2.0 * (state[:, 0] * state[:, 2] + state[:, 1] * state[:, 3])
    x1 = 2.0 * (state[:, 0] * state[:, 1] + state[:, 2] * state[:, 3])
    return _local_bloch_angles(
        torch.stack((z0, z1), dim=-1),
        torch.stack((x0, x1), dim=-1),
        eps,
    )


class RelationEvidenceSelector(nn.Module):
    """Bounded token evidence from a relation anchor and each token key."""

    selector_type = "base"

    def __init__(self, config: RelationEvidenceSelectorConfig) -> None:
        super().__init__()
        self.config = config
        register_qubits = (
            config.num_qubits // 2
            if config.evidence_readout == "connected_relation_token"
            else config.num_qubits
        )
        self.register_qubits = register_qubits
        token_projections = torch.stack(
            [
                _seeded_projection(
                    config.head_dim,
                    register_qubits,
                    config.seed + 17 * head,
                )
                for head in range(config.num_heads)
            ]
        )
        relation_projections = torch.stack(
            [
                _seeded_projection(
                    4 * config.head_dim,
                    register_qubits,
                    config.seed + 1009 + 19 * head,
                )
                for head in range(config.num_heads)
            ]
        )
        self.register_buffer("token_projections", token_projections)
        self.register_buffer("relation_projections", relation_projections)

        shape = (config.num_layers, config.num_heads, config.depth, register_qubits)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2003)
        self.data_scales = nn.Parameter(torch.ones(shape))
        self.data_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        if config.evidence_readout in {
            "factorized_observable",
            "connected_relation_token",
        }:
            self.relation_scales = nn.Parameter(torch.ones(shape))
            self.relation_biases = nn.Parameter(
                torch.empty(shape).uniform_(
                    -math.pi / 4,
                    math.pi / 4,
                    generator=generator,
                )
            )
        else:
            self.register_parameter("relation_scales", None)
            self.register_parameter("relation_biases", None)
        if config.evidence_readout == "connected_relation_token":
            observable_count = register_qubits * register_qubits
            if config.evidence_measurement_mode in {
                "relation_frame_bank",
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }:
                observable_count *= 2
            if config.evidence_correlation_mode == "dual_channel":
                observable_count *= 2
        else:
            observable_count = 2 * config.num_qubits
        observable_logits = torch.zeros(
            config.num_layers,
            config.num_heads,
            observable_count,
        )
        for head_index in range(config.num_heads):
            if config.evidence_measurement_mode in {
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }:
                frame_observables = observable_count // 2
                if config.evidence_correlation_mode == "dual_channel":
                    bank_observables = frame_observables // 2
                    selected = (5 * head_index) % bank_observables
                    for frame_index in range(2):
                        frame_start = frame_index * frame_observables
                        observable_logits[
                            :, head_index, frame_start + selected
                        ] = 1.0
                        observable_logits[
                            :,
                            head_index,
                            frame_start + bank_observables + selected,
                        ] = 1.0
                else:
                    selected = (5 * head_index) % frame_observables
                    observable_logits[:, head_index, selected] = 1.0
                    observable_logits[
                        :, head_index, frame_observables + selected
                    ] = 1.0
            else:
                observable_logits[
                    :, head_index, (5 * head_index) % observable_count
                ] = 1.0
        self.observable_logits = nn.Parameter(observable_logits)
        if config.evidence_task_readout == "dual":
            sufficiency_observable_logits = torch.zeros(
                config.num_layers,
                config.num_heads,
                2 * observable_count,
            )
            for head_index in range(config.num_heads):
                if config.evidence_measurement_mode in {
                    "relation_frame_bank",
                    "relation_frame_coherent",
                    "relation_directional",
                    "entanglement_directional",
                    "entanglement_phase_offset",
                }:
                    frame_observables = observable_count // 2
                    selected = (5 * head_index) % frame_observables
                    for frame_index in range(2):
                        frame_start = frame_index * 2 * frame_observables
                        sufficiency_observable_logits[
                            :, head_index, frame_start + selected
                        ] = 1.0
                else:
                    selected = (5 * head_index) % observable_count
                    sufficiency_observable_logits[:, head_index, selected] = 1.0
            self.sufficiency_observable_logits = nn.Parameter(
                sufficiency_observable_logits
            )
        else:
            self.register_parameter("sufficiency_observable_logits", None)
        if config.evidence_measurement_mode in {
            "relation_conditioned",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            self.raw_conditioning_gains = nn.Parameter(
                _raw_gain(
                    config.initial_conditioning_gain,
                    config.max_conditioning_gain,
                    (config.num_layers, config.num_heads),
                )
            )
        else:
            self.register_parameter("raw_conditioning_gains", None)
        if config.evidence_measurement_mode in {
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            self.raw_frame_fusion_gains = nn.Parameter(
                _raw_gain(
                    config.initial_frame_fusion_gain,
                    config.max_frame_fusion_gain,
                    (config.num_layers, config.num_heads),
                )
            )
        else:
            self.register_parameter("raw_frame_fusion_gains", None)
        if config.evidence_correlation_mode == "born_reliability":
            self.raw_reliability_exponents = nn.Parameter(
                torch.full(
                    (config.num_layers, config.num_heads),
                    math.log(math.expm1(1.0)),
                )
            )
        else:
            self.register_parameter("raw_reliability_exponents", None)
        self.raw_sharpness = nn.Parameter(
            torch.full(
                (config.num_layers, config.num_heads),
                math.log(math.expm1(config.initial_sharpness)),
            )
        )
        self.offsets = nn.Parameter(torch.zeros(config.num_layers, config.num_heads))
        if config.evidence_task_readout == "dual":
            self.raw_sufficiency_sharpness = nn.Parameter(
                torch.full(
                    (config.num_layers, config.num_heads),
                    math.log(math.expm1(config.initial_sharpness)),
                )
            )
        else:
            self.register_parameter("raw_sufficiency_sharpness", None)
        if config.intervention_mode == "direct_bias":
            self.raw_direct_gains = nn.Parameter(
                _raw_gain(
                    config.initial_direct_gain,
                    config.max_direct_gain,
                    (config.num_layers, config.num_heads),
                )
            )
        else:
            self.register_parameter("raw_direct_gains", None)
        self._capture_scores = False
        self._captured_scores: list[torch.Tensor] = []
        self._captured_steering_scores: list[torch.Tensor] = []
        self._captured_measurement_weights: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_relation_frame_angles: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_measurement_frame_contributions: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_correlation_channel_contributions: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_coherence_gates: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_reliability_gates: list[
            tuple[int, int, torch.Tensor]
        ] = []
        self._captured_quantum_diagnostics: list[
            tuple[int, int, dict[str, torch.Tensor]]
        ] = []
        self._measurement_frame_view = "full"

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "type": self.selector_type,
            "config": asdict(self.config),
        }
        if self.config.evidence_readout == "connected_relation_token":
            metadata["correlation_readout"] = {
                "mode": self.config.evidence_correlation_mode,
                "decomposition": list(EVIDENCE_CORRELATION_CHANNELS),
                "additional_measurement_circuits_per_frame": (
                    2
                    if self.config.evidence_correlation_mode
                    in {
                        "multiscale",
                        "correlation_gated",
                        "signed_gated",
                        "standardized_signed_gated",
                        "phase_selective",
                        "phase_rotated",
                        "dual_channel",
                        "born_reliability",
                    }
                    else 0
                ),
                "additional_qubits": 0,
                "additional_cross_register_cnots": 0,
            }
            if self.config.evidence_correlation_mode == "correlation_gated":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "connected_energy_gate",
                    "trainable_parameters": 0,
                }
            if self.config.evidence_correlation_mode == "signed_gated":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "signed_connected_energy_gate",
                    "trainable_parameters": 0,
                }
            if self.config.evidence_correlation_mode == "standardized_connected":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "pearson_connected_correlation",
                    "trainable_parameters": 0,
                    "free_amplitude": False,
                }
            if (
                self.config.evidence_correlation_mode
                == "standardized_signed_gated"
            ):
                metadata["correlation_readout"]["fusion"] = {
                    "type": "signed_pearson_residual",
                    "trainable_parameters": 0,
                    "free_amplitude": False,
                }
            if self.config.evidence_correlation_mode == "phase_selective":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "entangled_phase_orthogonal_residual",
                    "trainable_parameters": 0,
                    "free_amplitude": False,
                    "fixed_residual_scale": 0.5,
                }
            if self.config.evidence_correlation_mode == "phase_rotated":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "entangled_phase_half_angle_rotation",
                    "trainable_parameters": 0,
                    "free_amplitude": False,
                    "norm_preserving": True,
                }
            if self.config.evidence_correlation_mode == "dual_channel":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "independent_signed_observable_banks",
                    "classical_secondary_bank": "signed_square",
                    "observable_parameter_multiplier": 2.0,
                }
            if self.config.evidence_correlation_mode == "born_reliability":
                metadata["correlation_readout"]["fusion"] = {
                    "type": "born_energy_reliability_exponent",
                    "quality": "connected_over_connected_plus_post_product",
                    "trainable_parameters_per_layer_head": 1,
                    "free_amplitude": False,
                }
        if self.config.evidence_measurement_mode in {
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            metadata["measurement_resources"] = {
                "measurement_frames": 2,
                "observable_count": self.observable_logits.shape[-1],
                "relative_to_relation_frame": {
                    "additional_qubits": 0,
                    "additional_cross_register_cnots": 0,
                    "measurement_circuit_multiplier": 2.0,
                },
            }
            if self.config.evidence_measurement_mode in {
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }:
                metadata["measurement_resources"]["fusion"] = {
                    "type": "bounded_coherence_ratio",
                    "trainable_parameters_per_layer_head": 1,
                }
            if self.config.evidence_measurement_mode == "relation_directional":
                metadata["measurement_resources"]["conditioning"] = {
                    "type": "relation_conditioned_observable_axis",
                    "trainable_parameters_per_layer_head": 1,
                    "classical_control": "parameter_matched",
                }
            if self.config.evidence_measurement_mode == "entanglement_directional":
                metadata["measurement_resources"]["conditioning"] = {
                    "type": "entanglement_phase_observable_axis",
                    "signal": "balanced_reliability_cross_frame_phase",
                    "trainable_parameters_per_layer_head": 1,
                    "classical_control": "relation_axis_parameter_matched",
                    "zero_without_cross_register_entanglement": True,
                }
            if self.config.evidence_measurement_mode == "entanglement_phase_offset":
                metadata["measurement_resources"]["conditioning"] = {
                    "type": "phase_offset_entanglement_axis",
                    "signal": "balanced_reliability_phase_sine_cosine",
                    "parameter_role": "coupled_phase_and_amplitude",
                    "trainable_parameters_per_layer_head": 1,
                    "classical_control": "relation_axis_parameter_matched",
                    "zero_at_parameter_origin": True,
                    "zero_without_cross_register_entanglement": True,
                }
        if self.config.evidence_task_readout == "dual":
            if self.sufficiency_observable_logits is None:
                raise RuntimeError("dual task readout parameters are unavailable")
            if self.raw_sufficiency_sharpness is None:
                raise RuntimeError("dual task readout sharpness is unavailable")
            metadata["task_readout"] = {
                "mode": "dual",
                "shared_state_preparation": True,
                "steering_bank": "signed_phase_sensitive",
                "sufficiency_bank": "positive_connected_projectors",
                "additional_state_preparations_per_token": 0,
                "additional_measurement_circuits_per_frame": 0,
                "trainable_parameters": {
                    "steering_observable_bank": self.observable_logits.numel(),
                    "sufficiency_observable_bank": (
                        self.sufficiency_observable_logits.numel()
                    ),
                    "sufficiency_calibration": (
                        self.raw_sufficiency_sharpness.numel()
                    ),
                },
            }
        return metadata

    def observable_weights(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        return self._normalize_observable_weights(self.observable_logits[layer_index])

    def sufficiency_observable_weights(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.sufficiency_observable_logits is None:
            raise RuntimeError("sufficiency weights require dual task readout")
        logits = self.sufficiency_observable_logits[layer_index]
        if self.config.evidence_measurement_mode in {
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            frames = logits.reshape(logits.shape[:-1] + (2, -1))
            return torch.softmax(frames, dim=-1).reshape_as(logits)
        return torch.softmax(logits, dim=-1)

    def _normalize_observable_weights(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.evidence_weight_mode == "positive_simplex":
            return torch.softmax(logits, dim=-1)
        if self.config.evidence_measurement_mode in {
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            frames = logits.reshape(logits.shape[:-1] + (2, -1))
            centered = frames - frames.mean(dim=-1, keepdim=True)
            normalized = centered / centered.abs().sum(
                dim=-1, keepdim=True
            ).clamp_min(self.config.eps)
            return normalized.reshape_as(logits)
        centered = logits - logits.mean(dim=-1, keepdim=True)
        return centered / centered.abs().sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )

    def conditioning_gains(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.raw_conditioning_gains is None:
            raise RuntimeError(
                "conditioning gains require evidence_measurement_mode="
                "a relation- or entanglement-conditioned mode"
            )
        return self.config.max_conditioning_gain * torch.tanh(
            self.raw_conditioning_gains[layer_index]
        )

    def frame_fusion_gains(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.raw_frame_fusion_gains is None:
            raise RuntimeError(
                "frame fusion gains require evidence_measurement_mode="
                "a coherent two-frame mode"
            )
        return self.config.max_frame_fusion_gain * torch.tanh(
            self.raw_frame_fusion_gains[layer_index]
        )

    def reliability_exponents(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.raw_reliability_exponents is None:
            raise RuntimeError(
                "reliability exponents require evidence_correlation_mode="
                "'born_reliability'"
            )
        return F.softplus(self.raw_reliability_exponents[layer_index]) + self.config.eps

    def _reliability_frame_quality(
        self,
        correlation_channels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.selector_type == "quantum":
            signal = correlation_channels["connected"].reshape(
                correlation_channels["connected"].shape[0], 2, -1
            )
            reference = correlation_channels[
                "post_entanglement_product"
            ].reshape(signal.shape)
            signal_energy = torch.linalg.vector_norm(signal, dim=-1)
            reference_energy = torch.linalg.vector_norm(reference, dim=-1)
            return signal_energy / (signal_energy + reference_energy).clamp_min(
                self.config.eps
            )
        features = correlation_channels["pre_entanglement_product"].reshape(
            correlation_channels["pre_entanglement_product"].shape[0], 2, -1
        )
        centered = features - features.mean(dim=-1, keepdim=True)
        return torch.linalg.vector_norm(centered, dim=-1) / torch.linalg.vector_norm(
            features, dim=-1
        ).clamp_min(self.config.eps)

    def _entanglement_phase_components(
        self,
        correlation_channels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.selector_type != "quantum":
            raise RuntimeError(
                "entanglement direction features are only defined for quantum selectors"
            )
        signal = correlation_channels["standardized_connected"].reshape(
            correlation_channels["standardized_connected"].shape[0], 2, -1
        )
        reference = correlation_channels["pre_entanglement_product"].reshape(
            signal.shape
        )
        signal_norm = torch.linalg.vector_norm(signal, dim=1)
        reference_norm = torch.linalg.vector_norm(reference, dim=1)
        phase_sine = (
            signal[:, 0] * reference[:, 1]
            - signal[:, 1] * reference[:, 0]
        ) / (signal_norm * reference_norm + self.config.eps)
        phase_cosine = (
            signal[:, 0] * reference[:, 0]
            + signal[:, 1] * reference[:, 1]
        ) / (signal_norm * reference_norm + self.config.eps)
        reliability = (
            2.0
            * signal_norm
            / (signal_norm + reference_norm).clamp_min(self.config.eps)
        ).clamp(max=1.0)
        return (
            phase_sine.clamp(min=-1.0, max=1.0),
            phase_cosine.clamp(min=-1.0, max=1.0),
            reliability,
        )

    def _entanglement_direction_features(
        self,
        correlation_channels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        phase_sine, _phase_cosine, reliability = (
            self._entanglement_phase_components(correlation_channels)
        )
        direction = reliability * phase_sine
        return torch.cat((-direction, direction), dim=-1)

    def _entanglement_phase_offset_features(
        self,
        correlation_channels: dict[str, torch.Tensor],
        phase_offset: torch.Tensor,
    ) -> torch.Tensor:
        phase_sine, phase_cosine, reliability = (
            self._entanglement_phase_components(correlation_channels)
        )
        rotated_phase = (
            torch.cos(phase_offset) * phase_sine
            + torch.sin(phase_offset) * phase_cosine
        )
        direction = reliability * torch.sin(phase_offset) * rotated_phase
        return torch.cat((-direction, direction), dim=-1)

    def conditioned_observable_weights(
        self,
        relation_states: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        correlation_channels: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return sample-conditioned signed weights for connected observables."""
        if self.config.evidence_measurement_mode not in {
            "relation_conditioned",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            raise RuntimeError(
                "conditioned weights require evidence_measurement_mode="
                "a relation- or entanglement-conditioned mode"
            )
        self._validate_layer(layer_index)
        gain = self.conditioning_gains(layer_index)[head_index]
        apply_gain = True
        if (
            self.config.evidence_measurement_mode
            in {"entanglement_directional", "entanglement_phase_offset"}
            and self.selector_type == "quantum"
        ):
            if correlation_channels is None:
                raise RuntimeError(
                    "entanglement-conditioned weights require correlation channels"
                )
            if self.config.evidence_measurement_mode == "entanglement_phase_offset":
                conditioning_features = self._entanglement_phase_offset_features(
                    correlation_channels,
                    gain,
                )
                apply_gain = False
            else:
                conditioning_features = self._entanglement_direction_features(
                    correlation_channels
                )
        else:
            conditioning_features = self._relation_measurement_features(
                relation_states
            )
        base_logits = self.observable_logits[layer_index, head_index]
        if base_logits.shape[-1] == 2 * conditioning_features.shape[-1]:
            conditioning_features = conditioning_features.repeat(1, 2)
        if base_logits.shape[-1] != conditioning_features.shape[-1]:
            raise RuntimeError(
                "conditioned measurement features must match "
                "observable logits or be exactly half their width"
            )
        conditioned_logits = (
            conditioning_features if not apply_gain else gain * conditioning_features
        )
        return self._normalize_observable_weights(
            base_logits.unsqueeze(0) + conditioned_logits
        )

    def relation_frame_angles(self, relation_states: torch.Tensor) -> torch.Tensor:
        if self.config.evidence_measurement_mode not in {
            "relation_frame",
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            raise RuntimeError(
                "relation frame angles require evidence_measurement_mode="
                "'relation_frame', 'relation_frame_bank', or "
                "'relation_frame_coherent', 'relation_directional', or "
                "'entanglement_directional', or 'entanglement_phase_offset'"
            )
        return self.config.relation_frame_scale * self._relation_frame_angles(
            relation_states
        )

    def sharpness(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        return F.softplus(self.raw_sharpness[layer_index]) + self.config.eps

    def sufficiency_sharpness(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.raw_sufficiency_sharpness is None:
            raise RuntimeError("sufficiency sharpness requires dual task readout")
        return (
            F.softplus(self.raw_sufficiency_sharpness[layer_index])
            + self.config.eps
        )

    def direct_gains(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        if self.raw_direct_gains is None:
            raise RuntimeError("direct gains require intervention_mode='direct_bias'")
        return self.config.max_direct_gain * torch.tanh(
            self.raw_direct_gains[layer_index]
        )

    def direct_key_bias(
        self,
        scores: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        sufficiency_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Center context evidence into a direct per-head attention bias."""
        del sufficiency_scores
        if self.config.intervention_mode != "direct_bias":
            raise RuntimeError("direct_key_bias requires intervention_mode='direct_bias'")
        context = attention_mask & ~(subject_mask | object_mask)
        weights = context[:, None, :].to(device=scores.device, dtype=scores.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = scores - (scores * weights).sum(dim=-1, keepdim=True) / count
        gains = self.direct_gains(layer_index).view(1, -1, 1)
        if self.config.direct_bias_mode == "positive_excess":
            centered = F.relu(centered)
            gains = gains.abs()
        elif self.config.direct_bias_mode == "evidence_weighted_centered":
            centered = centered * scores
            gains = gains.abs()
        return centered * weights * gains

    def _validate_layer(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError(
                f"layer_index {layer_index} is outside [0, {self.config.num_layers})"
            )

    def _validate_inputs(
        self,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> None:
        if key.ndim != 4:
            raise ValueError("key must have shape (batch, num_heads, tokens, head_dim)")
        if key.shape[1] != self.config.num_heads or key.shape[3] != self.config.head_dim:
            raise ValueError("key dimensions do not match evidence-selector config")
        expected = (key.shape[0], key.shape[2])
        for name, mask in {
            "attention_mask": attention_mask,
            "subject_mask": subject_mask,
            "object_mask": object_mask,
        }.items():
            if mask.shape != expected:
                raise ValueError(f"{name} must match key batch and token dimensions")

    def _evidence_probabilities(
        self,
        logits: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        if self.config.evidence_gate_calibration == "none":
            return probabilities * attention_mask[:, None, :].to(probabilities.dtype)

        context = attention_mask & ~(subject_mask | object_mask)
        weights = context[:, None, :].to(device=logits.device, dtype=logits.dtype)
        count = weights.sum(dim=-1, keepdim=True)
        active = count > 0.0
        safe_count = count.clamp_min(1.0)
        target = logits.new_tensor(self.config.evidence_budget)
        target_logit = torch.log(target) - torch.log1p(-target)
        shift = (logits * weights).sum(dim=-1, keepdim=True) / safe_count
        shift = torch.where(active, shift - target_logit, torch.zeros_like(shift))

        # Newton projection preserves token ordering while fixing evidence mass.
        for _ in range(12):
            calibrated = torch.sigmoid(logits - shift)
            error = (calibrated * weights).sum(dim=-1, keepdim=True) / safe_count
            error = error - target
            slope = (
                calibrated * (1.0 - calibrated) * weights
            ).sum(dim=-1, keepdim=True) / safe_count
            update = error / slope.clamp_min(self.config.eps)
            shift = torch.where(active, (shift + update).clamp(-30.0, 30.0), shift)

        calibrated = torch.sigmoid(logits - shift)
        scores = torch.where(context[:, None, :], calibrated, probabilities)
        return scores * attention_mask[:, None, :].to(scores.dtype)

    def _encoding_inputs(
        self,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        head_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subject = _masked_head_mean(key, subject_mask, self.config.eps)
        object_ = _masked_head_mean(key, object_mask, self.config.eps)
        local = F.normalize(
            key[:, head_index].float(),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        if self.config.relation_anchor_mode == "entity_pair":
            relation = torch.cat(
                (subject, object_, subject - object_, subject * object_),
                dim=-1,
            )
            relation = relation[:, head_index, None, :].expand(-1, key.shape[2], -1)
        else:
            head_key = key[:, head_index].float()
            context = (
                attention_mask & ~(subject_mask | object_mask)
            ).to(device=head_key.device, dtype=head_key.dtype)[:, :, None]
            context_sum = (head_key * context).sum(dim=1, keepdim=True)
            context_count = context.sum(dim=1, keepdim=True)
            remaining_sum = context_sum - head_key * context
            remaining_count = context_count - context
            full_context = context_sum / context_count.clamp_min(1.0)
            leave_one_out = remaining_sum / remaining_count.clamp_min(1.0)
            context_anchor = torch.where(
                remaining_count > 0.0,
                leave_one_out,
                full_context.expand_as(leave_one_out),
            )
            subject_head = subject[:, head_index, None, :].expand_as(head_key)
            object_head = object_[:, head_index, None, :].expand_as(head_key)
            relation = torch.cat(
                (
                    subject_head,
                    object_head,
                    subject_head - object_head,
                    context_anchor,
                ),
                dim=-1,
            )
        anchor = F.normalize(
            relation.float(),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        return local, anchor

    def _feature_states(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _relation_measurement_features(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _relation_frame_angles(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _relation_token_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _connected_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> torch.Tensor:
        observables, _channels = self._relation_token_observable_features(
            token_states,
            relation_states,
        )
        return observables

    @contextmanager
    def use_measurement_frame_view(self, view: str) -> Iterator[None]:
        if view not in EVIDENCE_MEASUREMENT_FRAME_VIEWS:
            raise ValueError(
                "measurement frame view must be one of "
                f"{EVIDENCE_MEASUREMENT_FRAME_VIEWS}"
            )
        if (
            view != "full"
            and self.config.evidence_measurement_mode
            not in {
                "relation_frame_bank",
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }
        ):
            raise RuntimeError(
                "z/x measurement frame views require evidence_measurement_mode="
                "'relation_frame_bank' or 'relation_frame_coherent'"
            )
        previous = self._measurement_frame_view
        self._measurement_frame_view = view
        try:
            yield
        finally:
            self._measurement_frame_view = previous

    @contextmanager
    def capture_token_scores(self) -> Iterator[None]:
        if self._capture_scores:
            raise RuntimeError("token-evidence capture is already active")
        self._captured_scores.clear()
        self._captured_steering_scores.clear()
        self._captured_measurement_weights.clear()
        self._captured_relation_frame_angles.clear()
        self._captured_measurement_frame_contributions.clear()
        self._captured_correlation_channel_contributions.clear()
        self._captured_coherence_gates.clear()
        self._captured_reliability_gates.clear()
        self._captured_quantum_diagnostics.clear()
        self._capture_scores = True
        try:
            yield
        finally:
            self._capture_scores = False
            self._captured_scores.clear()
            self._captured_steering_scores.clear()
            self._captured_measurement_weights.clear()
            self._captured_relation_frame_angles.clear()
            self._captured_measurement_frame_contributions.clear()
            self._captured_correlation_channel_contributions.clear()
            self._captured_coherence_gates.clear()
            self._captured_reliability_gates.clear()
            self._captured_quantum_diagnostics.clear()

    def captured_token_scores(self) -> tuple[torch.Tensor, ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_scores)

    def captured_steering_scores(self) -> tuple[torch.Tensor, ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_steering_scores)

    def captured_measurement_weights(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_measurement_weights)

    def captured_relation_frame_angles(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_relation_frame_angles)

    def captured_measurement_frame_contributions(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_measurement_frame_contributions)

    def captured_correlation_channel_contributions(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_correlation_channel_contributions)

    def captured_coherence_gates(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_coherence_gates)

    def captured_reliability_gates(
        self,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_reliability_gates)

    def captured_quantum_diagnostics(
        self,
    ) -> tuple[tuple[int, int, dict[str, torch.Tensor]], ...]:
        if not self._capture_scores:
            raise RuntimeError("token-evidence capture is not active")
        return tuple(self._captured_quantum_diagnostics)

    def _sufficiency_observable_features(
        self,
        observables: torch.Tensor,
        correlation_channels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.selector_type == "quantum":
            source = correlation_channels.get("standardized_connected", observables)
        else:
            source = correlation_channels.get("pre_entanglement_product", observables)
        source = source.clamp(min=-1.0, max=1.0)
        if self.config.evidence_measurement_mode in {
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            frames = source.reshape(source.shape[0], 2, -1)
            positive = 0.5 * (1.0 + frames)
            negative = 0.5 * (1.0 - frames)
            return torch.cat((positive, negative), dim=-1).reshape(
                source.shape[0], -1
            )
        return torch.cat((0.5 * (1.0 + source), 0.5 * (1.0 - source)), dim=-1)

    def _positive_observable_expectation(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.evidence_measurement_mode in {
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            frame_features = features.reshape(features.shape[0], 2, -1)
            frame_weights = weights.reshape(2, -1)
            return (frame_features * frame_weights).sum(dim=-1).mean(dim=-1)
        return torch.sum(features * weights, dim=-1)

    def token_readouts(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return steering and counterfactual-sufficiency probabilities."""
        self._validate_layer(layer_index)
        self._validate_inputs(key, attention_mask, subject_mask, object_mask)
        batch, _heads, tokens, _dim = key.shape
        head_logits: list[torch.Tensor] = []
        sufficiency_head_logits: list[torch.Tensor] = []
        base_weights = self.observable_weights(layer_index)
        if self.config.evidence_task_readout == "dual":
            base_sufficiency_weights = self.sufficiency_observable_weights(layer_index)
        else:
            base_sufficiency_weights = None
        for head_index in range(self.config.num_heads):
            correlation_channels: dict[str, torch.Tensor] = {}
            local, anchor = self._encoding_inputs(
                key,
                attention_mask,
                subject_mask,
                object_mask,
                head_index,
            )
            if self.config.evidence_readout == "joint_observable":
                features = torch.cat((local, anchor), dim=-1)
                projection = torch.cat(
                    (
                        self.token_projections[head_index],
                        self.relation_projections[head_index],
                    ),
                    dim=0,
                )
                states = self._feature_states(
                    features.reshape(batch * tokens, -1),
                    projection,
                    layer_index=layer_index,
                    head_index=head_index,
                    branch="joint",
                )
                observables = self._observable_features(states)
            elif self.config.evidence_readout == "connected_relation_token":
                token_states = self._feature_states(
                    local.reshape(batch * tokens, -1),
                    self.token_projections[head_index],
                    layer_index=layer_index,
                    head_index=head_index,
                    branch="token",
                )
                relation_states = self._feature_states(
                    anchor.reshape(batch * tokens, -1),
                    self.relation_projections[head_index],
                    layer_index=layer_index,
                    head_index=head_index,
                    branch="relation",
                )
                if self.config.evidence_measurement_mode in {
                    "relation_frame",
                    "relation_frame_bank",
                    "relation_frame_coherent",
                    "relation_directional",
                    "entanglement_directional",
                    "entanglement_phase_offset",
                }:
                    frame_angles = self.relation_frame_angles(relation_states)
                    if self._capture_scores:
                        self._captured_relation_frame_angles.append(
                            (
                                layer_index,
                                head_index,
                                frame_angles.reshape(batch, tokens, -1),
                            )
                        )
                (
                    observables,
                    correlation_channels,
                ) = self._relation_token_observable_features(
                    token_states,
                    relation_states,
                )
                if self.config.evidence_correlation_mode == "born_reliability":
                    quality = self._reliability_frame_quality(
                        correlation_channels
                    )
                    exponent = self.reliability_exponents(layer_index)[head_index]
                    powered = quality.clamp_min(self.config.eps).pow(exponent)
                    reliability = torch.where(
                        quality > self.config.eps,
                        powered,
                        torch.zeros_like(powered),
                    )
                    observables = (
                        observables.reshape(observables.shape[0], 2, -1)
                        * reliability.unsqueeze(-1)
                    ).reshape(observables.shape[0], -1)
                    correlation_channels = dict(correlation_channels)
                    correlation_channels["born_reliability"] = observables
                    if self._capture_scores:
                        self._captured_reliability_gates.append(
                            (
                                layer_index,
                                head_index,
                                torch.stack((quality, reliability), dim=-1).reshape(
                                    batch, tokens, 2, 2
                                ),
                            )
                        )
                if self.config.evidence_measurement_mode in {
                    "relation_conditioned",
                    "relation_directional",
                    "entanglement_directional",
                    "entanglement_phase_offset",
                }:
                    weights = self.conditioned_observable_weights(
                        relation_states,
                        layer_index=layer_index,
                        head_index=head_index,
                        correlation_channels=correlation_channels,
                    )
                    if self._capture_scores:
                        self._captured_measurement_weights.append(
                            (
                                layer_index,
                                head_index,
                                weights.reshape(batch, tokens, -1),
                            )
                        )
                else:
                    weights = base_weights[head_index]
            else:
                token_states = self._feature_states(
                    local.reshape(batch * tokens, -1),
                    self.token_projections[head_index],
                    layer_index=layer_index,
                    head_index=head_index,
                    branch="token",
                )
                relation_states = self._feature_states(
                    anchor.reshape(batch * tokens, -1),
                    self.relation_projections[head_index],
                    layer_index=layer_index,
                    head_index=head_index,
                    branch="relation",
                )
                observables = self._observable_features(
                    token_states
                ) * self._observable_features(relation_states)
                weights = base_weights[head_index]
            if self.config.evidence_readout == "joint_observable":
                weights = base_weights[head_index]
            if self.config.evidence_measurement_mode in {
                "relation_frame_bank",
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }:
                if base_sufficiency_weights is not None:
                    sufficiency_features = self._sufficiency_observable_features(
                        observables,
                        correlation_channels,
                    )
                    sufficiency_expectation = self._positive_observable_expectation(
                        sufficiency_features,
                        base_sufficiency_weights[head_index],
                    )
                    sufficiency_head_logits.append(
                        (
                            self.sufficiency_sharpness(layer_index)[head_index]
                            * sufficiency_expectation
                        ).reshape(batch, tokens)
                    )
                frame_weights = weights.reshape(weights.shape[:-1] + (2, -1))
                raw_frame_contributions = (
                    observables.reshape(observables.shape[0], 2, -1)
                    * frame_weights
                ).sum(dim=-1)
                channel_frame_contributions = None
                if correlation_channels:
                    channel_frame_contributions = torch.stack(
                        [
                            (
                                correlation_channels[channel].reshape(
                                    observables.shape[0], 2, -1
                                )
                                * frame_weights
                            ).sum(dim=-1)
                            for channel in EVIDENCE_CORRELATION_CHANNELS
                        ],
                        dim=-2,
                    )
                if self.config.evidence_measurement_mode in {
                    "relation_frame_coherent",
                    "relation_directional",
                    "entanglement_directional",
                    "entanglement_phase_offset",
                }:
                    frame_observables = observables.reshape(
                        observables.shape[0], 2, -1
                    )
                    frame_energy = torch.linalg.vector_norm(
                        frame_observables,
                        dim=-1,
                    )
                    coherence_ratio = frame_energy[:, 1] / frame_energy.sum(
                        dim=-1
                    ).clamp_min(self.config.eps)
                    effective_gate = (
                        self.frame_fusion_gains(layer_index)[head_index]
                        * coherence_ratio
                    )
                    raw_frame_contributions = torch.stack(
                        (
                            raw_frame_contributions[:, 0],
                            effective_gate * raw_frame_contributions[:, 1],
                        ),
                        dim=-1,
                    )
                    if channel_frame_contributions is not None:
                        frame_gates = torch.stack(
                            (torch.ones_like(effective_gate), effective_gate),
                            dim=-1,
                        )
                        channel_frame_contributions = (
                            channel_frame_contributions
                            * frame_gates.unsqueeze(-2)
                        )
                    if self._capture_scores:
                        self._captured_coherence_gates.append(
                            (
                                layer_index,
                                head_index,
                                torch.stack(
                                    (coherence_ratio, effective_gate), dim=-1
                                ).reshape(batch, tokens, 2),
                            )
                        )
                frame_contributions = raw_frame_contributions.reshape(
                    batch, tokens, 2
                )
                if self._capture_scores:
                    self._captured_measurement_frame_contributions.append(
                        (
                            layer_index,
                            head_index,
                            frame_contributions,
                        )
                    )
                    if channel_frame_contributions is not None:
                        self._captured_correlation_channel_contributions.append(
                            (
                                layer_index,
                                head_index,
                                channel_frame_contributions.reshape(
                                    batch,
                                    tokens,
                                    len(EVIDENCE_CORRELATION_CHANNELS),
                                    2,
                                ),
                            )
                        )
                flat_frame_contributions = frame_contributions.reshape(
                    batch * tokens, 2
                )
                if self._measurement_frame_view == "z":
                    expectation = flat_frame_contributions[:, 0]
                elif self._measurement_frame_view == "x":
                    expectation = flat_frame_contributions[:, 1]
                else:
                    expectation = flat_frame_contributions.sum(dim=-1)
            else:
                if base_sufficiency_weights is not None:
                    sufficiency_features = self._sufficiency_observable_features(
                        observables,
                        correlation_channels,
                    )
                    sufficiency_expectation = self._positive_observable_expectation(
                        sufficiency_features,
                        base_sufficiency_weights[head_index],
                    )
                    sufficiency_head_logits.append(
                        (
                            self.sufficiency_sharpness(layer_index)[head_index]
                            * sufficiency_expectation
                        ).reshape(batch, tokens)
                    )
                expectation = torch.sum(observables * weights, dim=-1)
            logits = (
                self.sharpness(layer_index)[head_index] * expectation
                + self.offsets[layer_index, head_index]
            )
            head_logits.append(logits.reshape(batch, tokens))
        steering_logits = torch.stack(head_logits, dim=1)
        if base_sufficiency_weights is None:
            steering_scores = self._evidence_probabilities(
                steering_logits,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
            sufficiency_scores = steering_scores
        else:
            steering_scores = torch.sigmoid(steering_logits)
            steering_scores = steering_scores * attention_mask[:, None, :].to(
                steering_scores.dtype
            )
            sufficiency_scores = self._evidence_probabilities(
                torch.stack(sufficiency_head_logits, dim=1),
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
        if self._capture_scores:
            self._captured_steering_scores.append(steering_scores)
            self._captured_scores.append(sufficiency_scores)
        return steering_scores, sufficiency_scores

    def token_scores(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the evidence probabilities used for counterfactual views."""
        return self.token_readouts(
            key,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )[1]

    def steering_residual(
        self,
        centered: torch.Tensor,
        steering_evidence: torch.Tensor,
        sufficiency_evidence: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the legacy bounded multiplicative evidence modulation."""
        del sufficiency_evidence, subject_mask, object_mask
        key_mask = attention_mask[:, None, None, :].to(centered.dtype)
        modulated = centered * (2.0 * steering_evidence[:, :, None, :])
        key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        modulated = modulated - (
            (modulated * key_mask).sum(dim=-1, keepdim=True) / key_count
        )
        if query_mask is None:
            query_mask = attention_mask
        active_queries = query_mask[:, None, :, None].to(centered.dtype)
        return modulated * active_queries * key_mask

    def permuted_context_scores(
        self,
        scores: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        seed: int,
        detach: bool = False,
    ) -> torch.Tensor:
        """Randomly relocate the same evidence mass among valid context tokens."""
        source = scores.detach() if detach else scores
        shuffled = source.clone()
        context = attention_mask & ~(subject_mask | object_mask)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for batch_index in range(scores.shape[0]):
            indices = torch.nonzero(context[batch_index], as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            for head_index in range(scores.shape[1]):
                permutation = torch.randperm(indices.numel(), generator=generator)
                target = indices.to(device=scores.device)
                source_indices = indices[permutation].to(device=scores.device)
                shuffled[batch_index, head_index, target] = source[
                    batch_index,
                    head_index,
                    source_indices,
                ]
        return shuffled

    def _view_scores(
        self,
        scores: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.evidence_view_score_mode == "positive":
            return scores

        context = attention_mask & ~(subject_mask | object_mask)
        weights = context[:, None, :].to(device=scores.device, dtype=scores.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        center = (scores * weights).sum(dim=-1, keepdim=True) / count
        positive = F.relu(scores - center) / (1.0 - center).clamp_min(self.config.eps)
        negative = F.relu(center - scores) / center.clamp_min(self.config.eps)
        magnitude = (positive + negative).clamp(max=1.0)
        return magnitude * weights

    def view_weights(
        self,
        scores: torch.Tensor,
        *,
        view: str,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        random_seed: int = 0,
        detach_random: bool = False,
        steering_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del steering_scores
        if view not in EVIDENCE_VIEW_CHOICES:
            raise ValueError(f"view must be one of {EVIDENCE_VIEW_CHOICES}")
        if view == "full":
            return torch.ones_like(scores)
        gate = self._view_scores(
            scores,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )
        if view.startswith("random_"):
            gate = self.permuted_context_scores(
                gate,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
                seed=random_seed,
                detach=detach_random,
            )
        if view.endswith("drop"):
            gate = 1.0 - gate
        soft_context = self.config.mask_floor + (1.0 - self.config.mask_floor) * gate
        entity = subject_mask | object_mask
        context = attention_mask & ~entity
        return torch.where(
            entity[:, None, :],
            torch.ones_like(scores),
            torch.where(context[:, None, :], soft_context, torch.ones_like(scores)),
        )


class QuantumRelationEvidenceSelector(RelationEvidenceSelector):
    """Entangled Born-observable token-evidence selector."""

    selector_type = "quantum"

    def __init__(self, config: RelationEvidenceSelectorConfig) -> None:
        super().__init__(config)
        basis = torch.arange(2**config.num_qubits, dtype=torch.long)
        masks = [1 << qubit for qubit in range(config.num_qubits)]
        masks.extend(
            (1 << qubit) | (1 << ((qubit + 1) % config.num_qubits))
            for qubit in range(config.num_qubits)
        )
        signs: list[torch.Tensor] = []
        for mask in masks:
            parity = torch.zeros_like(basis)
            for bit in range(config.num_qubits):
                if mask & (1 << bit):
                    parity = parity.bitwise_xor((basis >> bit).bitwise_and(1))
            signs.append(1.0 - 2.0 * parity.float())
        self.register_buffer("observable_signs", torch.stack(signs), persistent=False)

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(states.square(), self.observable_signs.transpose(0, 1))

    def _relation_measurement_features(self, states: torch.Tensor) -> torch.Tensor:
        return _two_qubit_pauli_features(states)

    def _relation_frame_angles(self, states: torch.Tensor) -> torch.Tensor:
        return _two_qubit_local_bloch_angles(states, self.config.eps)

    def _relation_token_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        product_state = _join_register_states(relation_states, token_states)
        state = product_state
        if self.config.cross_entanglement:
            state = _cross_register_entangle(state, self.register_qubits)
        channel_frames = {
            channel: [] for channel in EVIDENCE_CORRELATION_CHANNELS
        }

        def append_channels(
            product_frame_state: torch.Tensor,
            frame_state: torch.Tensor,
        ) -> None:
            (
                _product_joint,
                pre_entanglement_product,
                _product_connected,
            ) = _relation_token_z_correlations(
                product_frame_state,
                self.register_qubits,
            )
            (
                _joint,
                post_entanglement_product,
                connected,
                relation_expectations,
                token_expectations,
            ) = _relation_token_z_statistics(
                frame_state,
                self.register_qubits,
            )
            relation_variance = (1.0 - relation_expectations.square()).clamp_min(
                0.0
            )
            token_variance = (1.0 - token_expectations.square()).clamp_min(0.0)
            correlation_variance = (
                relation_variance.unsqueeze(-1) * token_variance.unsqueeze(-2)
            ).reshape_as(connected)
            # Clamp before sqrt so zero-variance states have a finite gradient.
            correlation_scale = torch.sqrt(
                correlation_variance.clamp_min(self.config.eps)
            )
            if self.config.cross_entanglement:
                standardized_connected = (connected / correlation_scale).clamp(
                    min=-1.0,
                    max=1.0,
                )
            else:
                standardized_connected = torch.zeros_like(connected)
            channel_frames["pre_entanglement_product"].append(
                pre_entanglement_product
            )
            channel_frames["post_entanglement_product"].append(
                post_entanglement_product
            )
            channel_frames["connected"].append(connected)
            channel_frames["total"].append(
                post_entanglement_product + connected
            )
            channel_frames["multiscale"].append(
                pre_entanglement_product + connected
            )
            correlation_gate = connected.square() / (
                connected.square()
                + pre_entanglement_product.square()
                + self.config.eps
            )
            channel_frames["correlation_gated"].append(
                connected + correlation_gate * pre_entanglement_product
            )
            signed_gate = connected * pre_entanglement_product / (
                connected.square()
                + pre_entanglement_product.square()
                + self.config.eps
            )
            channel_frames["signed_gated"].append(
                connected + signed_gate * pre_entanglement_product
            )
            channel_frames["standardized_connected"].append(
                standardized_connected
            )
            standardized_signed_gate = (
                standardized_connected * pre_entanglement_product
                / (
                    standardized_connected.square()
                    + pre_entanglement_product.square()
                    + self.config.eps
                )
            )
            channel_frames["standardized_signed_gated"].append(
                standardized_connected
                + standardized_signed_gate * pre_entanglement_product
            )
            channel_frames["phase_selective"].append(
                standardized_connected
            )
            channel_frames["phase_rotated"].append(
                standardized_connected
            )
            channel_frames["dual_channel"].append(
                connected + pre_entanglement_product
            )
            channel_frames["born_reliability"].append(
                connected + signed_gate * pre_entanglement_product
            )

        if self.config.evidence_measurement_mode in {
            "relation_frame",
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            angles = self.relation_frame_angles(relation_states)
            total_qubits = 2 * self.register_qubits
            frame_offsets = (0.0, math.pi / 2)
            if self.config.evidence_measurement_mode == "relation_frame":
                frame_offsets = frame_offsets[:1]
            for offset in frame_offsets:
                frame_state = state
                product_frame_state = product_state
                relation_offset = (
                    0.0
                    if self.config.evidence_measurement_mode
                    in {
                        "relation_frame_coherent",
                        "relation_directional",
                        "entanglement_directional",
                        "entanglement_phase_offset",
                    }
                    else offset
                )
                for qubit in range(self.register_qubits):
                    relation_rotation = -(angles[:, qubit] + relation_offset)
                    token_rotation = -(angles[:, qubit] + offset)
                    frame_state = _apply_ry(
                        frame_state,
                        relation_rotation,
                        qubit,
                        total_qubits,
                    )
                    frame_state = _apply_ry(
                        frame_state,
                        token_rotation,
                        self.register_qubits + qubit,
                        total_qubits,
                    )
                    product_frame_state = _apply_ry(
                        product_frame_state,
                        relation_rotation,
                        qubit,
                        total_qubits,
                    )
                    product_frame_state = _apply_ry(
                        product_frame_state,
                        token_rotation,
                        self.register_qubits + qubit,
                        total_qubits,
                    )
                append_channels(product_frame_state, frame_state)
            channels = {
                name: torch.cat(values, dim=-1)
                for name, values in channel_frames.items()
            }
            signal_frames = channels["standardized_connected"].reshape(
                channels["standardized_connected"].shape[0], 2, -1
            )
            baseline_frames = channels["pre_entanglement_product"].reshape(
                channels["pre_entanglement_product"].shape[0], 2, -1
            )
            signal_norm = torch.linalg.vector_norm(signal_frames, dim=1)
            baseline_norm = torch.linalg.vector_norm(baseline_frames, dim=1)
            determinant = (
                signal_frames[:, 0] * baseline_frames[:, 1]
                - signal_frames[:, 1] * baseline_frames[:, 0]
            )
            phase_sine = (
                determinant / (signal_norm * baseline_norm + self.config.eps)
            ).clamp(min=-1.0, max=1.0)
            phase_cosine = (
                (
                    signal_frames[:, 0] * baseline_frames[:, 0]
                    + signal_frames[:, 1] * baseline_frames[:, 1]
                )
                / (signal_norm * baseline_norm + self.config.eps)
            ).clamp(min=-1.0, max=1.0)
            baseline_orthogonal = torch.stack(
                (-baseline_frames[:, 1], baseline_frames[:, 0]),
                dim=1,
            ) / baseline_norm.unsqueeze(1).clamp_min(self.config.eps)
            phase_residual = (
                0.5
                * phase_sine.unsqueeze(1)
                * signal_norm.unsqueeze(1)
                * baseline_orthogonal
            )
            channels["phase_selective"] = (
                channels["standardized_signed_gated"].reshape(
                    channels["standardized_signed_gated"].shape[0], 2, -1
                )
                + phase_residual
            ).reshape_as(channels["standardized_signed_gated"])
            half_cosine = torch.sqrt(
                ((1.0 + phase_cosine) * 0.5).clamp_min(0.0)
            )
            half_sine = phase_sine.sign() * torch.sqrt(
                ((1.0 - phase_cosine) * 0.5).clamp_min(0.0)
            )
            base_frames = channels["standardized_signed_gated"].reshape(
                channels["standardized_signed_gated"].shape[0], 2, -1
            )
            channels["phase_rotated"] = torch.stack(
                (
                    half_cosine * base_frames[:, 0]
                    - half_sine * base_frames[:, 1],
                    half_sine * base_frames[:, 0]
                    + half_cosine * base_frames[:, 1],
                ),
                dim=1,
            ).reshape_as(channels["standardized_signed_gated"])
            if self.config.evidence_correlation_mode == "dual_channel":
                zeros = torch.zeros_like(channels["connected"])
                connected_bank = _interleave_frame_banks(
                    channels["connected"], zeros
                )
                pre_bank = _interleave_frame_banks(
                    zeros, channels["pre_entanglement_product"]
                )
                post_bank = _interleave_frame_banks(
                    zeros, channels["post_entanglement_product"]
                )
                channels = {
                    "pre_entanglement_product": pre_bank,
                    "post_entanglement_product": post_bank,
                    "connected": connected_bank,
                    "total": connected_bank + post_bank,
                    "multiscale": connected_bank + pre_bank,
                    "correlation_gated": _interleave_frame_banks(
                        channels["correlation_gated"], zeros
                    ),
                    "signed_gated": _interleave_frame_banks(
                        channels["signed_gated"], zeros
                    ),
                    "standardized_connected": _interleave_frame_banks(
                        channels["standardized_connected"], zeros
                    ),
                    "standardized_signed_gated": _interleave_frame_banks(
                        channels["standardized_signed_gated"], zeros
                    ),
                    "phase_selective": _interleave_frame_banks(
                        channels["phase_selective"], zeros
                    ),
                    "phase_rotated": _interleave_frame_banks(
                        channels["phase_rotated"], zeros
                    ),
                    "dual_channel": connected_bank + pre_bank,
                    "born_reliability": _interleave_frame_banks(
                        channels["born_reliability"], zeros
                    ),
                }
            return channels[self.config.evidence_correlation_mode], channels
        append_channels(product_state, state)
        channels = {name: values[0] for name, values in channel_frames.items()}
        return channels[self.config.evidence_correlation_mode], channels

    def _feature_states(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        if branch in {"joint", "token"}:
            scales = self.data_scales[layer_index, head_index]
            biases = self.data_biases[layer_index, head_index]
        elif branch == "relation":
            if self.relation_scales is None or self.relation_biases is None:
                raise RuntimeError("factorized relation parameters are unavailable")
            scales = self.relation_scales[layer_index, head_index]
            biases = self.relation_biases[layer_index, head_index]
        else:
            raise ValueError("branch must be 'joint', 'token', or 'relation'")
        return _data_reuploading_state(
            features,
            projection,
            scales,
            biases,
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        )


class QuantumNESSRelationEvidenceSelector(QuantumRelationEvidenceSelector):
    """Quantum necessity/sufficiency evidence with non-complementary readouts."""

    selector_type = "qness"

    def __init__(self, config: RelationEvidenceSelectorConfig) -> None:
        if config.evidence_readout != "connected_relation_token":
            raise ValueError("Q-NESS requires connected_relation_token readout")
        if config.evidence_task_readout != "dual":
            raise ValueError("Q-NESS requires evidence_task_readout='dual'")
        if config.evidence_measurement_mode != "fixed":
            raise ValueError("Q-NESS uses its own fixed noncommuting observable pair")
        if config.evidence_gate_calibration != "none":
            raise ValueError("Q-NESS does not use complementary budget calibration")
        if config.num_qubits != 4:
            raise ValueError("Q-NESS currently requires exactly four qubits")
        super().__init__(config)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 9001)
        phase_signs = torch.randint(
            0,
            2,
            (2**config.num_qubits,),
            generator=generator,
            dtype=torch.long,
        ).mul(2).sub(1).to(dtype=torch.float32)
        self.register_buffer("phase_scramble_signs", phase_signs)

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata["qness"] = {
            "control": self.config.qness_control,
            "necessity_observable": "connected_Z_relation_Z_token",
            "sufficiency_observable": "X_relation_Z_token",
            "commutator": "[Z_r Z_t, X_r Z_t] != 0",
            "non_complementary_readouts": True,
            "drop_signal": "sigmoid(-logit(necessity))",
            "phase_scramble_seed": self.config.seed + 9001,
        }
        metadata["task_readout"] = {
            "mode": "qness",
            "shared_state_preparation": True,
            "necessity_bank": "signed_connected_ZZ",
            "sufficiency_bank": "positive_XZ_projectors",
            "complement_calibration": False,
        }
        return metadata

    def _qness_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        product_state = _join_register_states(relation_states, token_states)
        state = product_state
        entangled = self.config.cross_entanglement and (
            self.config.qness_control != "separable"
        )
        if entangled:
            state = _cross_register_entangle(state, self.register_qubits)
        if self.config.qness_control == "phase_scrambled":
            state = state * self.phase_scramble_signs.to(
                device=state.device,
                dtype=state.dtype,
            )

        _joint, _product, necessity, _relation, _token = (
            _relation_token_z_statistics(state, self.register_qubits)
        )
        if self.config.qness_control == "commuting":
            sufficiency = necessity
        elif self.config.qness_control == "dephased":
            sufficiency = torch.zeros_like(necessity)
        else:
            sufficiency = _relation_x_token_z_correlations(
                state,
                self.register_qubits,
            )

        diagnostic_state = state.detach()
        if self._capture_scores and self.config.quantum_diagnostic_limit > 0:
            diagnostic_state = diagnostic_state[: self.config.quantum_diagnostic_limit]
        with torch.no_grad():
            if self.config.qness_control == "dephased":
                off_diagonal = torch.zeros(
                    diagnostic_state.shape[0],
                    device=state.device,
                    dtype=state.dtype,
                )
                mutual_information = torch.zeros_like(off_diagonal)
            else:
                off_diagonal = _pure_state_off_diagonal_norm(diagnostic_state)
                mutual_information = _pure_state_register_mutual_information(
                    diagnostic_state,
                    self.register_qubits,
                    self.config.eps,
                )
        def expand(values: torch.Tensor) -> torch.Tensor:
            expanded = state.new_zeros((state.shape[0],))
            expanded[: values.shape[0]] = values.to(device=state.device, dtype=state.dtype)
            return expanded

        commutator_norm = state.new_full(
            (state.shape[0],),
            0.0 if self.config.qness_control == "commuting" else 2.0,
        )
        diagnostics = {
            "off_diagonal_density_norm": expand(off_diagonal),
            "mutual_information": expand(mutual_information),
            "observable_commutator_norm": commutator_norm,
            "entangled_state": state.new_full(
                (state.shape[0],),
                1.0 if entangled else 0.0,
            ),
        }
        return necessity, sufficiency, diagnostics

    def _qness_delta_from_readouts(
        self,
        base_scores: torch.Tensor,
        necessity_scores: torch.Tensor,
        sufficiency_scores: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Implement delta_j = tau_N(n_j-E_a[n]) + tau_S(s_j-E_a[s])."""
        if necessity_scores.shape != sufficiency_scores.shape:
            raise ValueError("Q-NESS readouts must have matching shapes")
        safe_necessity = necessity_scores.clamp(self.config.eps, 1.0 - self.config.eps)
        safe_sufficiency = sufficiency_scores.clamp(
            self.config.eps,
            1.0 - self.config.eps,
        )
        necessity_logits = torch.logit(safe_necessity)
        sufficiency_logits = torch.logit(safe_sufficiency)
        if base_scores.ndim == 4:
            valid = key_mask[:, None, None, :]
            attention = torch.softmax(
                base_scores.masked_fill(~valid, -torch.inf),
                dim=-1,
            )
            necessity = necessity_logits[:, :, None, :]
            sufficiency = sufficiency_logits[:, :, None, :]
            delta = (
                necessity - (attention * necessity).sum(dim=-1, keepdim=True)
                + sufficiency
                - (attention * sufficiency).sum(dim=-1, keepdim=True)
            )
            return delta * valid.to(delta.dtype)
        if base_scores.ndim == 3:
            valid = key_mask[:, None, :].to(base_scores.dtype)
            attention = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return (
                necessity_logits
                - (attention * necessity_logits).sum(dim=-1, keepdim=True)
                + sufficiency_logits
                - (attention * sufficiency_logits).sum(dim=-1, keepdim=True)
            ) * valid
        raise ValueError("base_scores must have shape (B,H,Q,T) or (B,H,T)")

    def steering_residual(
        self,
        centered: torch.Tensor,
        steering_evidence: torch.Tensor,
        sufficiency_evidence: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self._qness_delta_from_readouts(
            centered,
            steering_evidence,
            sufficiency_evidence,
            attention_mask,
        )
        if query_mask is None:
            query_mask = attention_mask
        del subject_mask, object_mask
        return delta * query_mask[:, None, :, None].to(delta.dtype)

    def direct_key_bias(
        self,
        scores: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        sufficiency_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sufficiency_scores is None:
            raise ValueError("Q-NESS direct bias requires both independent readouts")
        context = attention_mask & ~(subject_mask | object_mask)
        delta = self._qness_delta_from_readouts(
            torch.zeros_like(scores),
            scores,
            sufficiency_scores,
            context,
        )
        return delta * self.direct_gains(layer_index).view(1, -1, 1)

    def view_weights(
        self,
        scores: torch.Tensor,
        *,
        view: str,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        random_seed: int = 0,
        detach_random: bool = False,
        steering_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steering_scores is None:
            raise ValueError("Q-NESS views require necessity and sufficiency readouts")
        if view not in EVIDENCE_VIEW_CHOICES:
            raise ValueError(f"view must be one of {EVIDENCE_VIEW_CHOICES}")
        if view == "full":
            return torch.ones_like(scores)
        gate = scores if view.endswith("keep") else torch.sigmoid(
            -torch.logit(
                steering_scores.clamp(self.config.eps, 1.0 - self.config.eps)
            )
        )
        gate = self._view_scores(
            gate,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )
        if view.startswith("random_"):
            gate = self.permuted_context_scores(
                gate,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
                seed=random_seed,
                detach=detach_random,
            )
        soft_context = self.config.mask_floor + (1.0 - self.config.mask_floor) * gate
        entity = subject_mask | object_mask
        context = attention_mask & ~entity
        return torch.where(
            entity[:, None, :],
            torch.ones_like(scores),
            torch.where(context[:, None, :], soft_context, torch.ones_like(scores)),
        )

    def token_readouts(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer(layer_index)
        self._validate_inputs(key, attention_mask, subject_mask, object_mask)
        batch, _heads, tokens, _dim = key.shape
        necessity_logits: list[torch.Tensor] = []
        sufficiency_logits: list[torch.Tensor] = []
        necessity_weights = self.observable_weights(layer_index)
        sufficiency_weights = self.sufficiency_observable_weights(layer_index)
        for head_index in range(self.config.num_heads):
            local, anchor = self._encoding_inputs(
                key,
                attention_mask,
                subject_mask,
                object_mask,
                head_index,
            )
            token_states = self._feature_states(
                local.reshape(batch * tokens, -1),
                self.token_projections[head_index],
                layer_index=layer_index,
                head_index=head_index,
                branch="token",
            )
            relation_states = self._feature_states(
                anchor.reshape(batch * tokens, -1),
                self.relation_projections[head_index],
                layer_index=layer_index,
                head_index=head_index,
                branch="relation",
            )
            necessity, sufficiency, diagnostics = self._qness_observable_features(
                token_states,
                relation_states,
            )
            n_expectation = torch.sum(
                necessity * necessity_weights[head_index], dim=-1
            )
            positive_sufficiency = torch.cat(
                (0.5 * (1.0 + sufficiency), 0.5 * (1.0 - sufficiency)),
                dim=-1,
            )
            s_expectation = torch.sum(
                positive_sufficiency * sufficiency_weights[head_index], dim=-1
            )
            necessity_logits.append(
                (
                    self.sharpness(layer_index)[head_index] * n_expectation
                    + self.offsets[layer_index, head_index]
                ).reshape(batch, tokens)
            )
            sufficiency_logits.append(
                (
                    self.sufficiency_sharpness(layer_index)[head_index]
                    * s_expectation
                ).reshape(batch, tokens)
            )
            if self._capture_scores:
                self._captured_quantum_diagnostics.append(
                    (
                        layer_index,
                        head_index,
                        {
                            name: value.reshape(batch, tokens)
                            for name, value in diagnostics.items()
                        },
                    )
                )
        necessity_scores = torch.sigmoid(torch.stack(necessity_logits, dim=1))
        sufficiency_scores = torch.sigmoid(torch.stack(sufficiency_logits, dim=1))
        attention = attention_mask[:, None, :].to(necessity_scores.dtype)
        necessity_scores = necessity_scores * attention
        sufficiency_scores = sufficiency_scores * attention
        if self._capture_scores:
            self._captured_steering_scores.append(necessity_scores)
            self._captured_scores.append(sufficiency_scores)
        return necessity_scores, sufficiency_scores


class ClassicalNESSRelationEvidenceSelector(QuantumNESSRelationEvidenceSelector):
    """Parameter-matched shared-trunk classical dual-head Q-NESS control."""

    selector_type = "qness_classical"

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata["qness"]["classical_control"] = "shared_trunk_dual_head"
        metadata["qness"]["commutator"] = "not_applicable_classical_control"
        return metadata

    def _feature_states(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        if branch in {"joint", "token"}:
            scales = self.data_scales[layer_index, head_index]
            biases = self.data_biases[layer_index, head_index]
        elif branch == "relation":
            if self.relation_scales is None or self.relation_biases is None:
                raise RuntimeError("factorized relation parameters are unavailable")
            scales = self.relation_scales[layer_index, head_index]
            biases = self.relation_biases[layer_index, head_index]
        else:
            raise ValueError("branch must be 'joint', 'token', or 'relation'")
        normalized = F.normalize(features.float(), p=2, dim=-1, eps=self.config.eps)
        angles = self.config.angle_scale * torch.matmul(normalized, projection)
        phase = angles
        for depth_index in range(self.config.depth):
            phase = torch.sin(
                phase + angles * scales[depth_index] + biases[depth_index]
            )
        state = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        return F.normalize(state, p=2, dim=-1, eps=self.config.eps)

    def _qness_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        qubits = self.register_qubits
        token_z = token_states[..., :qubits].square() - token_states[..., qubits:].square()
        relation_z = (
            relation_states[..., :qubits].square()
            - relation_states[..., qubits:].square()
        )
        relation_x = 2.0 * (
            relation_states[..., :qubits] * relation_states[..., qubits:]
        )
        necessity = (
            relation_z.unsqueeze(-1) * token_z.unsqueeze(-2)
        ).reshape(token_states.shape[0], -1)
        sufficiency = (
            relation_x.unsqueeze(-1) * token_z.unsqueeze(-2)
        ).reshape(token_states.shape[0], -1)
        zero = token_states.new_zeros(token_states.shape[0])
        diagnostics = {
            "off_diagonal_density_norm": zero,
            "mutual_information": zero,
            "observable_commutator_norm": zero,
            "entangled_state": zero,
        }
        return necessity, sufficiency, diagnostics


class ClassicalRelationEvidenceSelector(RelationEvidenceSelector):
    """Parameter-matched separable trigonometric evidence control."""

    selector_type = "classical"

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        qubits = self.config.num_qubits
        local_z = qubits * (
            states[..., :qubits].square() - states[..., qubits:].square()
        )
        adjacent_zz = local_z * torch.roll(local_z, shifts=-1, dims=-1)
        return torch.cat((local_z, adjacent_zz), dim=-1)

    def _relation_measurement_features(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.shape[-1] != 4:
            raise ValueError("matched classical relation states must have shape (batch, 4)")
        cosine = states[:, :2]
        sine = states[:, 2:]
        local_z = 2.0 * (cosine.square() - sine.square())
        local_x = 4.0 * cosine * sine
        return torch.stack(
            (
                local_z[:, 0],
                local_z[:, 1],
                local_z[:, 0] * local_z[:, 1],
                local_x[:, 0] * local_x[:, 1],
            ),
            dim=-1,
        )

    def _relation_frame_angles(self, states: torch.Tensor) -> torch.Tensor:
        qubits = self.register_qubits
        if states.ndim != 2 or states.shape[-1] != 2 * qubits:
            raise ValueError(
                "matched classical relation states must have shape "
                f"(batch, {2 * qubits})"
            )
        cosine = states[:, :qubits]
        sine = states[:, qubits:]
        return _local_bloch_angles(
            cosine.square() - sine.square(),
            2.0 * cosine * sine,
            self.config.eps,
        )

    def _relation_token_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        qubits = self.register_qubits
        token_local = token_states[..., :qubits].square() - token_states[..., qubits:].square()
        relation_local = (
            relation_states[..., :qubits].square()
            - relation_states[..., qubits:].square()
        )
        if self.config.evidence_measurement_mode in {
            "relation_frame",
            "relation_frame_bank",
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            angles = self.relation_frame_angles(relation_states)
            token_x = 2.0 * (
                token_states[..., :qubits]
                * token_states[..., qubits:]
            )
            relation_x = 2.0 * (
                relation_states[..., :qubits]
                * relation_states[..., qubits:]
            )
            cosine = torch.cos(angles)
            sine = torch.sin(angles)
            token_frame = cosine * token_local + sine * token_x
            relation_frame = cosine * relation_local + sine * relation_x
            frame_features = [
                (relation_frame.unsqueeze(-1) * token_frame.unsqueeze(-2)).reshape(
                    token_states.shape[0], -1
                )
            ]
            if self.config.evidence_measurement_mode in {
                "relation_frame_bank",
                "relation_frame_coherent",
                "relation_directional",
                "entanglement_directional",
                "entanglement_phase_offset",
            }:
                token_orthogonal = -sine * token_local + cosine * token_x
                relation_orthogonal = -sine * relation_local + cosine * relation_x
                frame_features.append(
                    (
                        relation_orthogonal.unsqueeze(-1)
                        * token_orthogonal.unsqueeze(-2)
                    ).reshape(token_states.shape[0], -1)
                )
            features = torch.cat(frame_features, dim=-1)
        else:
            features = (
                relation_local.unsqueeze(-1) * token_local.unsqueeze(-2)
            ).reshape(token_states.shape[0], -1)
        channels = {
            "pre_entanglement_product": features,
            "post_entanglement_product": features,
            "connected": torch.zeros_like(features),
            "total": features,
            "multiscale": features,
            "correlation_gated": torch.zeros_like(features),
            "signed_gated": torch.zeros_like(features),
            "standardized_connected": torch.zeros_like(features),
            "standardized_signed_gated": torch.zeros_like(features),
            "phase_selective": torch.zeros_like(features),
            "phase_rotated": torch.zeros_like(features),
            "dual_channel": features,
            "born_reliability": features,
        }
        if self.config.evidence_correlation_mode == "dual_channel":
            return _classical_dual_channel_features(features)
        return features, channels

    def _feature_states(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        if branch in {"joint", "token"}:
            scales = self.data_scales[layer_index, head_index]
            biases = self.data_biases[layer_index, head_index]
        elif branch == "relation":
            if self.relation_scales is None or self.relation_biases is None:
                raise RuntimeError("factorized relation parameters are unavailable")
            scales = self.relation_scales[layer_index, head_index]
            biases = self.relation_biases[layer_index, head_index]
        else:
            raise ValueError("branch must be 'joint', 'token', or 'relation'")
        normalized = F.normalize(features.float(), p=2, dim=-1, eps=self.config.eps)
        angles = self.config.angle_scale * torch.matmul(normalized, projection)
        phase = angles
        for depth_index in range(self.config.depth):
            phase = torch.sin(
                phase
                + angles * scales[depth_index]
                + biases[depth_index]
            )
        state = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        return F.normalize(state, p=2, dim=-1, eps=self.config.eps)


class StrongClassicalRelationEvidenceSelector(ClassicalRelationEvidenceSelector):
    """Nondegenerate relation-aligned classical two-frame control."""

    selector_type = "classical_strong"

    def _relation_token_observable_features(
        self,
        token_states: torch.Tensor,
        relation_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.evidence_measurement_mode not in {
            "relation_frame_coherent",
            "relation_directional",
            "entanglement_directional",
            "entanglement_phase_offset",
        }:
            return super()._relation_token_observable_features(
                token_states,
                relation_states,
            )
        qubits = self.register_qubits
        token_z = (
            token_states[..., :qubits].square()
            - token_states[..., qubits:].square()
        )
        relation_z = (
            relation_states[..., :qubits].square()
            - relation_states[..., qubits:].square()
        )
        token_x = 2.0 * (
            token_states[..., :qubits] * token_states[..., qubits:]
        )
        relation_x = 2.0 * (
            relation_states[..., :qubits] * relation_states[..., qubits:]
        )
        angles = self.relation_frame_angles(relation_states)
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        relation_aligned = cosine * relation_z + sine * relation_x
        token_aligned = cosine * token_z + sine * token_x
        token_orthogonal = -sine * token_z + cosine * token_x
        z_features = (
            relation_aligned.unsqueeze(-1) * token_aligned.unsqueeze(-2)
        ).reshape(token_states.shape[0], -1)
        x_features = (
            relation_aligned.unsqueeze(-1) * token_orthogonal.unsqueeze(-2)
        ).reshape(token_states.shape[0], -1)
        features = torch.cat((z_features, x_features), dim=-1)
        channels = {
            "pre_entanglement_product": features,
            "post_entanglement_product": features,
            "connected": torch.zeros_like(features),
            "total": features,
            "multiscale": features,
            "correlation_gated": torch.zeros_like(features),
            "signed_gated": torch.zeros_like(features),
            "standardized_connected": torch.zeros_like(features),
            "standardized_signed_gated": torch.zeros_like(features),
            "phase_selective": torch.zeros_like(features),
            "phase_rotated": torch.zeros_like(features),
            "dual_channel": features,
            "born_reliability": features,
        }
        if self.config.evidence_correlation_mode == "dual_channel":
            return _classical_dual_channel_features(features)
        return features, channels


def build_relation_evidence_selector(
    selector_type: str,
    config: RelationEvidenceSelectorConfig,
) -> RelationEvidenceSelector:
    if selector_type == "quantum":
        return QuantumRelationEvidenceSelector(config)
    qness_controls = {
        "qness": "none",
        "qness_commuting": "commuting",
        "qness_separable": "separable",
        "qness_phase_scrambled": "phase_scrambled",
        "qness_dephased": "dephased",
    }
    if selector_type == "qness_classical":
        qness_config = replace(
            config,
            evidence_readout="connected_relation_token",
            evidence_task_readout="dual",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="fixed",
            evidence_gate_calibration="none",
            qness_control="none",
        )
        return ClassicalNESSRelationEvidenceSelector(qness_config)
    if selector_type in qness_controls:
        requested_control = (
            config.qness_control
            if selector_type == "qness"
            else qness_controls[selector_type]
        )
        qness_config = replace(
            config,
            evidence_readout="connected_relation_token",
            evidence_task_readout="dual",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="fixed",
            evidence_gate_calibration="none",
            qness_control=requested_control,
        )
        return QuantumNESSRelationEvidenceSelector(qness_config)
    if selector_type == "classical":
        return ClassicalRelationEvidenceSelector(config)
    if selector_type == "classical_strong":
        return StrongClassicalRelationEvidenceSelector(config)
    raise ValueError(f"selector_type must be one of {EVIDENCE_SELECTOR_TYPES}")
