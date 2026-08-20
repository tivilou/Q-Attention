"""Quantum connected-correlation consensus witness for unordered key pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any

import torch
import torch.nn as nn


QCCW_KERNEL_TYPES = ("quantum", "product", "bilinear")


@dataclass(frozen=True)
class ConnectedConsensusWitnessConfig:
    num_candidates: int
    head_dim: int
    num_key_pairs: int = 15
    angle_scale: float = 1.0
    seed: int = 7331
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_candidates <= 0 or self.head_dim <= 0:
            raise ValueError("num_candidates and head_dim must be positive")
        expected_pairs = self.num_key_pairs
        if expected_pairs != len(tuple(combinations(range(6), 2))):
            raise ValueError("Stage-0 QCCW requires the six-key, fifteen-pair task")
        if self.angle_scale <= 0.0 or self.eps <= 0.0:
            raise ValueError("angle_scale and eps must be positive")


def _product_state(angles: torch.Tensor) -> torch.Tensor:
    state = torch.ones(angles.shape[0], 1, device=angles.device, dtype=angles.dtype)
    for qubit in range(angles.shape[1]):
        local = torch.stack(
            (torch.cos(angles[:, qubit] / 2.0), torch.sin(angles[:, qubit] / 2.0)),
            dim=-1,
        )
        state = (state.unsqueeze(-1) * local.unsqueeze(1)).reshape(angles.shape[0], -1)
    return state


def _apply_cz(state: torch.Tensor, control: int, target: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    control_mask = 1 << (num_qubits - control - 1)
    target_mask = 1 << (num_qubits - target - 1)
    phase = torch.where(
        ((indices & control_mask) != 0) & ((indices & target_mask) != 0),
        -torch.ones_like(indices, dtype=state.dtype),
        torch.ones_like(indices, dtype=state.dtype),
    )
    return state * phase


def _apply_x(state: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    mask = 1 << (num_qubits - qubit - 1)
    return state[:, indices ^ mask]


def _expectation_x(state: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    return (state * _apply_x(state, qubit, num_qubits)).sum(dim=-1)


def _expectation_xx(state: torch.Tensor, left: int, right: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    left_mask = 1 << (num_qubits - left - 1)
    right_mask = 1 << (num_qubits - right - 1)
    return (state * state[:, indices ^ left_mask ^ right_mask]).sum(dim=-1)


class ConnectedConsensusWitness(nn.Module):
    """Swap-symmetric two-register connected-XX scorer.

    The forward scoring path accepts only observable query/key tensors.  Target
    labels and evidence pairs are used by the Stage-0 runner's training loss,
    never by this module.
    """

    kernel_type = "quantum"
    entangled = True

    def __init__(self, config: ConnectedConsensusWitnessConfig, candidate_frames: torch.Tensor) -> None:
        super().__init__()
        if candidate_frames.shape != (config.num_candidates, config.head_dim, config.head_dim):
            raise ValueError("candidate_frames must have shape (candidates, dim, dim)")
        self.config = config
        self.register_buffer("candidate_frames", candidate_frames.detach().float())
        self.register_buffer(
            "pair_indices",
            torch.tensor(tuple(combinations(range(6), 2)), dtype=torch.long),
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.scales = nn.Parameter(torch.ones(2))
        self.biases = nn.Parameter(torch.empty(2).uniform_(-math.pi / 6, math.pi / 6, generator=generator))

    def _local_features(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or key.ndim != 4:
            raise ValueError("query must be (batch, queries, dim), key must be (batch, queries, keys, dim)")
        if query.shape[0:2] != key.shape[0:2] or query.shape[-1] != self.config.head_dim:
            raise ValueError("query and key dimensions do not match QCCW")
        if key.shape[2] != 6:
            raise ValueError("QCCW Stage-0 requires six candidate keys")
        if key_second is None:
            key_second = key
        if key_second.shape != key.shape:
            raise ValueError("key_second must have the same shape as key")
        frames = self.candidate_frames.to(device=query.device, dtype=query.dtype)
        transformed_query = torch.einsum("bqd,cdh->bqch", query, frames)
        pairs = self.pair_indices.to(query.device)
        key_a = key[:, :, pairs[:, 0], :]
        key_b = key_second[:, :, pairs[:, 1], :]
        # Two scalar, candidate-conditioned local features are shared by both registers.
        compat_a = torch.einsum("bqcd,bqpd->bqcp", transformed_query, key_a)
        compat_b = torch.einsum("bqcd,bqpd->bqcp", transformed_query, key_b)
        distance_a = -(key_a[:, :, None, :, :] - transformed_query[:, :, :, None, :]).square().mean(dim=-1)
        distance_b = -(key_b[:, :, None, :, :] - transformed_query[:, :, :, None, :]).square().mean(dim=-1)
        features_a = torch.stack((compat_a, distance_a), dim=-1)
        features_b = torch.stack((compat_b, distance_b), dim=-1)
        return features_a, features_b

    def _connected_score(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        flat_a = features_a.reshape(-1, 2)
        flat_b = features_b.reshape(-1, 2)
        angles_a = self.config.angle_scale * flat_a * self.scales + self.biases
        angles_b = self.config.angle_scale * flat_b * self.scales + self.biases
        # Interleaved registers: (a0, b0, a1, b1), with tied local encoders.
        angles = torch.stack(
            (angles_a[:, 0], angles_b[:, 0], angles_a[:, 1], angles_b[:, 1]), dim=-1
        )
        state = _product_state(angles + math.pi / 2.0)
        if self.entangled:
            state = _apply_cz(state, 0, 1, 4)
            state = _apply_cz(state, 2, 3, 4)
        xa = _expectation_x(state, 0, 4)
        xb = _expectation_x(state, 1, 4)
        xc = _expectation_x(state, 2, 4)
        xd = _expectation_x(state, 3, 4)
        xx_ab = _expectation_xx(state, 0, 1, 4)
        xx_cd = _expectation_xx(state, 2, 3, 4)
        connected = 0.5 * ((xx_ab - xa * xb) + (xx_cd - xc * xd))
        return connected.reshape(features_a.shape[:-1])

    def pair_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features_a, features_b = self._local_features(query, key, key_second=key_second)
        return self._connected_score(features_a, features_b)

    def candidate_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_second: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_scores = self.pair_scores(query, key, key_second=key_second)
        candidate_scores, pair_choice = pair_scores.max(dim=-1)
        return candidate_scores, pair_choice

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "config": asdict(self.config),
            "plugin": {
                "id": f"qccw_{self.kernel_type}",
                "version": "0.1.0",
                "type": "standalone_unordered_pair_connected_correlator",
                "insertion_point": "candidate_pair_scoring_before_bounded_attention_residual",
                "hypothesis": "connected XX over two tied key registers identifies relation-consistent key pairs",
                "input_schema": "query[batch,query,dim],key[batch,query,6,dim],candidate_frames",
                "output_schema": "pair_score[batch,query,candidate,15]",
                "requires": ["fixed_label_free_error_witness", "candidate_frame_bank"],
                "conflicts": ["oracle_pair_inference", "gold_label_inference"],
                "deterministic": True,
                "resource_estimate": {
                    "data_qubits": 4,
                    "ancilla_qubits": 0,
                    "depth": 2,
                    "two_qubit_gates_per_pair": 2 if self.entangled else 0,
                    "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
                    "shots": "not modeled by exact statevector Stage-0",
                },
                "failure_signatures": [
                    "connected score is numerically trivial",
                    "pair-consistency AUC below 0.75",
                    "matched bilinear control wins",
                    "entangler cut or key shuffle preserves held-out gain",
                    "non-finite score or gradient",
                ],
            },
        }


class ProductConnectedConsensusWitness(ConnectedConsensusWitness):
    kernel_type = "product"
    entangled = False


class BilinearConnectedConsensusWitness(ConnectedConsensusWitness):
    kernel_type = "bilinear"
    entangled = False

    def __init__(self, config: ConnectedConsensusWitnessConfig, candidate_frames: torch.Tensor) -> None:
        super().__init__(config, candidate_frames)
        del self.scales
        del self.biases
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.bilinear = nn.Parameter(torch.eye(2) + 0.05 * torch.randn(2, 2, generator=generator))

    def _connected_score(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        matrix = 0.5 * (self.bilinear + self.bilinear.transpose(0, 1))
        flat_a = features_a.reshape(-1, 2)
        flat_b = features_b.reshape(-1, 2)
        score = torch.einsum("ni,ij,nj->n", flat_a, matrix, flat_b)
        return score.reshape(features_a.shape[:-1])


def build_connected_consensus_witness(
    kernel_type: str,
    config: ConnectedConsensusWitnessConfig,
    candidate_frames: torch.Tensor,
) -> ConnectedConsensusWitness:
    if kernel_type == "quantum":
        return ConnectedConsensusWitness(config, candidate_frames)
    if kernel_type == "product":
        return ProductConnectedConsensusWitness(config, candidate_frames)
    if kernel_type == "bilinear":
        return BilinearConnectedConsensusWitness(config, candidate_frames)
    raise ValueError(f"kernel_type must be one of {QCCW_KERNEL_TYPES}")


def unordered_pair_index(pair: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
    """Map unordered key pairs to the explicit fifteen-pair allowlist."""
    if pair.shape[-1] != 2:
        raise ValueError("pair must end with two key indices")
    normalized = torch.sort(pair, dim=-1).values
    matches = (normalized[..., None, :] == pair_indices.to(pair.device)).all(dim=-1)
    if not matches.any(dim=-1).all():
        raise ValueError("pair contains a key outside the explicit pair allowlist")
    return matches.to(torch.long).argmax(dim=-1)
