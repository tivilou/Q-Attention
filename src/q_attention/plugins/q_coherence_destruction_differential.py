"""Coherent-minus-dephased phase-parity witness for unordered key pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
import torch.nn as nn

from q_attention.plugins.q_connected_consensus_witness import (
    ConnectedConsensusWitness,
    ConnectedConsensusWitnessConfig,
    _apply_cz,
    _expectation_x,
    _expectation_xx,
    _product_state,
)


QCDD_KERNEL_TYPES = ("quantum", "product", "sincos")


@dataclass(frozen=True)
class CoherenceDifferentialConfig(ConnectedConsensusWitnessConfig):
    """QCDD shares the frozen four-parameter QCCW state preparation."""


def _apply_y(state: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    mask = 1 << (num_qubits - qubit - 1)
    negative_i = torch.full(
        (2**num_qubits,), -1j, dtype=state.dtype, device=state.device
    )
    positive_i = -negative_i
    phase = torch.where((indices & mask) == 0, negative_i, positive_i)
    return state[:, indices ^ mask] * phase


def _expectation_y_string(
    state: torch.Tensor,
    qubits: Sequence[int],
    num_qubits: int,
) -> torch.Tensor:
    transformed = state
    for qubit in qubits:
        transformed = _apply_y(transformed, int(qubit), num_qubits)
    return (state.conj() * transformed).sum(dim=-1).real


def dephase_density_matrix(
    state: torch.Tensor,
    qubits: Sequence[int],
    *,
    num_qubits: int,
) -> torch.Tensor:
    """Apply complete computational-basis dephasing to selected qubits."""
    if state.ndim != 2 or state.shape[-1] != 2**num_qubits:
        raise ValueError("state must have shape (batch, 2**num_qubits)")
    density = state[:, :, None] * state[:, None, :].conj()
    indices = torch.arange(2**num_qubits, device=state.device)
    keep = torch.ones(2**num_qubits, 2**num_qubits, dtype=torch.bool, device=state.device)
    for qubit in qubits:
        mask = 1 << (num_qubits - int(qubit) - 1)
        bits = (indices & mask) != 0
        keep &= bits[:, None] == bits[None, :]
    return density * keep.to(density.dtype)


def pauli_string_operator(
    paulis: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    matrices = {
        "I": torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
    }
    if not paulis or any(pauli not in matrices for pauli in paulis):
        raise ValueError("paulis must be a non-empty I/X/Y/Z string")
    result = matrices[paulis[0]]
    for pauli in paulis[1:]:
        result = torch.kron(result, matrices[pauli])
    return result


def density_expectation(density: torch.Tensor, paulis: str) -> torch.Tensor:
    if density.ndim != 3 or density.shape[-1] != density.shape[-2]:
        raise ValueError("density must have shape (batch, state_dim, state_dim)")
    operator = pauli_string_operator(
        paulis, dtype=density.dtype, device=density.device
    )
    if operator.shape != density.shape[-2:]:
        raise ValueError("Pauli string length does not match density dimension")
    return torch.einsum("bij,ji->b", density, operator).real


def explicit_dephased_connected_yyyy(state: torch.Tensor) -> torch.Tensor:
    """Reference density-matrix evaluation for the analytic QCDD null."""
    density = dephase_density_matrix(state, (1, 3), num_qubits=4)
    four_body = density_expectation(density, "YYYY")
    register_a = density_expectation(density, "YIYI")
    register_b = density_expectation(density, "IYIY")
    return four_body - register_a * register_b


class CoherenceDestructionDifferential(ConnectedConsensusWitness):
    """Four-body phase parity removed by dephasing the second key register."""

    kernel_type = "quantum"
    entangled = True

    def __init__(
        self,
        config: CoherenceDifferentialConfig,
        candidate_frames: torch.Tensor,
    ) -> None:
        super().__init__(config, candidate_frames)

    def _joint_state(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        flat_a = features_a.reshape(-1, 2)
        flat_b = features_b.reshape(-1, 2)
        angles_a = self.config.angle_scale * flat_a * self.scales + self.biases
        angles_b = self.config.angle_scale * flat_b * self.scales + self.biases
        angles = torch.stack(
            (
                angles_a[:, 0],
                angles_b[:, 0],
                angles_a[:, 1],
                angles_b[:, 1],
            ),
            dim=-1,
        )
        state = _product_state(angles + math.pi / 2.0).to(torch.complex64)
        if self.entangled:
            state = _apply_cz(state, 0, 1, 4)
            state = _apply_cz(state, 2, 3, 4)
        return state

    @staticmethod
    def _state_components(state: torch.Tensor) -> dict[str, torch.Tensor]:
        yyyy = _expectation_y_string(state, (0, 1, 2, 3), 4)
        register_a_yy = _expectation_y_string(state, (0, 2), 4)
        register_b_yy = _expectation_y_string(state, (1, 3), 4)
        coherent_yyyy = yyyy - register_a_yy * register_b_yy

        y0 = _expectation_y_string(state, (0,), 4)
        y1 = _expectation_y_string(state, (1,), 4)
        y2 = _expectation_y_string(state, (2,), 4)
        y3 = _expectation_y_string(state, (3,), 4)
        yy01 = _expectation_y_string(state, (0, 1), 4) - y0 * y1
        yy23 = _expectation_y_string(state, (2, 3), 4) - y2 * y3

        real_state = state.real
        x0 = _expectation_x(real_state, 0, 4)
        x1 = _expectation_x(real_state, 1, 4)
        x2 = _expectation_x(real_state, 2, 4)
        x3 = _expectation_x(real_state, 3, 4)
        xx01 = _expectation_xx(real_state, 0, 1, 4) - x0 * x1
        xx23 = _expectation_xx(real_state, 2, 3, 4) - x2 * x3
        return {
            "coherent_yyyy": coherent_yyyy,
            # Every term contains Y on the dephased key-b register, so the
            # density-matrix result is exactly zero. A unit test checks this
            # analytic path against explicit dephasing.
            "dephased_yyyy": torch.zeros_like(coherent_yyyy),
            "differential": coherent_yyyy,
            "secondary_connected_yy": 0.5 * (yy01 + yy23),
            "raw_qccw_xx": 0.5 * (xx01 + xx23),
            "yyyy_moment": yyyy,
            "register_a_yy_moment": register_a_yy,
            "register_b_yy_moment": register_b_yy,
        }

    def pair_score_components(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features_a, features_b = self._local_features(
            query, key, key_second=key_second
        )
        state = self._joint_state(features_a, features_b)
        flat_components = self._state_components(state)
        shape = features_a.shape[:-1]
        return {name: value.reshape(shape) for name, value in flat_components.items()}

    def pair_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pair_score_components(
            query, key, key_second=key_second
        )["differential"]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "config": asdict(self.config),
            "plugin": {
                "id": f"qcdd_{self.kernel_type}",
                "version": "0.1.0",
                "type": "standalone_unordered_pair_coherence_differential",
                "insertion_point": "joint_pair_state_readout_before_candidate_aggregation",
                "hypothesis": (
                    "coherent-minus-dephased four-body phase parity ranks "
                    "relation-consistent pairs beyond a rank-two trigonometric control"
                ),
                "input_schema": "query[batch,query,dim],key[batch,query,6,dim],candidate_frames",
                "output_schema": "pair_score[batch,query,candidate,15] with named readout components",
                "requires": ["candidate_frame_bank", "exact_unordered_pair_enumeration"],
                "conflicts": ["gold_pair_inference", "attention_action_before_readout_gate"],
                "deterministic": True,
                "resource_estimate": {
                    "data_qubits": 4,
                    "ancilla_qubits": 0,
                    "depth": 2,
                    "two_qubit_gates_per_pair": 2 if self.entangled else 0,
                    "trainable_parameter_count": sum(
                        parameter.numel() for parameter in self.parameters()
                    ),
                    "dephasing": "analytic null verified against explicit density matrix",
                    "shots": "estimated by the Stage-0 runner; not executed",
                },
                "failure_signatures": [
                    "differential is numerically trivial",
                    "held-out pair AUC below 0.75",
                    "sine/cosine control comes within 0.02 AUC",
                    "key shuffle preserves pair AUC",
                    "estimated shots exceed 4096",
                ],
            },
        }


class ProductCoherenceDifferential(CoherenceDestructionDifferential):
    kernel_type = "product"
    entangled = False


class SineCosinePairControl(CoherenceDestructionDifferential):
    kernel_type = "sincos"
    entangled = False

    def __init__(
        self,
        config: CoherenceDifferentialConfig,
        candidate_frames: torch.Tensor,
    ) -> None:
        super().__init__(config, candidate_frames)
        del self.scales
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.weights = nn.Parameter(torch.ones(2))
        self.biases.data.uniform_(-math.pi / 6, math.pi / 6, generator=generator)

    def pair_score_components(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features_a, features_b = self._local_features(
            query, key, key_second=key_second
        )
        angles_a = features_a * self.weights + self.biases
        angles_b = features_b * self.weights + self.biases
        score = 0.5 * (
            torch.sin(angles_a[..., 0]) * torch.sin(angles_b[..., 0])
            + torch.cos(angles_a[..., 1]) * torch.cos(angles_b[..., 1])
        )
        zeros = torch.zeros_like(score)
        return {
            "coherent_yyyy": zeros,
            "dephased_yyyy": zeros,
            "differential": score,
            "secondary_connected_yy": zeros,
            "raw_qccw_xx": zeros,
            "yyyy_moment": zeros,
            "register_a_yy_moment": zeros,
            "register_b_yy_moment": zeros,
        }


def build_coherence_differential(
    kernel_type: str,
    config: CoherenceDifferentialConfig,
    candidate_frames: torch.Tensor,
) -> CoherenceDestructionDifferential:
    if kernel_type == "quantum":
        return CoherenceDestructionDifferential(config, candidate_frames)
    if kernel_type == "product":
        return ProductCoherenceDifferential(config, candidate_frames)
    if kernel_type == "sincos":
        return SineCosinePairControl(config, candidate_frames)
    raise ValueError(f"kernel_type must be one of {QCDD_KERNEL_TYPES}")
