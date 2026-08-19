"""Candidate-all quantum fields for the consensus error-witness task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from q_attention.plugins.q_margin_credit import _product_reuploading_state
from q_attention.plugins.quantum_steering import (
    _data_reuploading_state,
    _seeded_projection,
)


CONSENSUS_QUANTUM_KERNEL_TYPES = ("quantum", "classical")


@dataclass(frozen=True)
class ConsensusQuantumEstimatorConfig:
    num_candidates: int
    head_dim: int
    register_qubits: int = 3
    depth: int = 2
    angle_scale: float = 1.0
    seed: int = 7331
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(
            self.num_candidates,
            self.head_dim,
            self.register_qubits,
            self.depth,
        ) <= 0:
            raise ValueError("estimator dimensions must be positive")
        if self.angle_scale <= 0.0 or self.eps <= 0.0:
            raise ValueError("angle_scale and eps must be positive")


class ConsensusQuantumEstimator(nn.Module):
    """Signed candidate-relative evidence field with a matched control."""

    kernel_type = "base"
    entangled_state = False

    def __init__(
        self,
        config: ConsensusQuantumEstimatorConfig,
        candidate_frames: torch.Tensor,
    ) -> None:
        super().__init__()
        if candidate_frames.shape != (
            config.num_candidates,
            config.head_dim,
            config.head_dim,
        ):
            raise ValueError("candidate_frames must have shape (candidates, dim, dim)")
        self.config = config
        self.state_dim = 2**config.register_qubits
        feature_dim = 3 * config.head_dim
        self.register_buffer("candidate_frames", candidate_frames.detach().float())
        self.register_buffer(
            "parity_observable",
            torch.tensor(
                [
                    -1.0 if value.bit_count() % 2 else 1.0
                    for value in range(self.state_dim)
                ],
                dtype=torch.float32,
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        projection = _seeded_projection(feature_dim, config.register_qubits, config.seed)
        self.input_projection = nn.Parameter(projection)
        self.scales = nn.Parameter(torch.ones(config.depth, config.register_qubits))
        self.biases = nn.Parameter(
            torch.empty(config.depth, config.register_qubits).uniform_(
                -math.pi / 6, math.pi / 6, generator=generator
            )
        )
        self._last_field: torch.Tensor | None = None

    def _state(self, features: torch.Tensor) -> torch.Tensor:
        builder = (
            _data_reuploading_state
            if self.entangled_state
            else _product_reuploading_state
        )
        return builder(
            features,
            self.input_projection,
            self.scales,
            self.biases,
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        )

    def field(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 4:
            raise ValueError("query must be (batch, queries, dim), key must be (batch, queries, keys, dim)")
        if query.shape[0:2] != key.shape[0:2] or query.shape[-1] != self.config.head_dim:
            raise ValueError("query and key dimensions do not match the estimator")
        batch, queries, keys, _ = key.shape
        frames = self.candidate_frames.to(device=query.device, dtype=query.dtype)
        transformed_query = torch.einsum("bqd,cdh->bqch", query, frames)
        fields = []
        observable = self.parity_observable.to(device=query.device, dtype=query.dtype)
        for candidate in range(self.config.num_candidates):
            candidate_query = transformed_query[:, :, candidate, :]
            features = torch.cat(
                (
                    query[:, :, None, :].expand(-1, -1, keys, -1),
                    key,
                    candidate_query[:, :, None, :].expand(-1, -1, keys, -1),
                ),
                dim=-1,
            )
            state = self._state(features.reshape(batch * queries * keys, -1))
            expectation = (state.square() * observable).sum(dim=-1)
            fields.append(expectation.reshape(batch, queries, keys))
        result = torch.stack(fields, dim=2)
        self._last_field = result.detach()
        return result

    def candidate_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        field = self.field(query, key)
        return field.topk(2, dim=-1).values.mean(dim=-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "config": asdict(self.config),
            "plugin": {
                "id": f"q_consensus_quantum_estimator_{self.kernel_type}",
                "version": "0.1.0",
                "type": "standalone_query_local_candidate_field",
                "insertion_point": "pre_softmax_attention_score_action_selector",
                "hypothesis": (
                    "a signed parity observable over query, key, and candidate-transformed "
                    "query can learn candidate-relative two-key consensus without labels at inference"
                ),
                "input_schema": "query[batch,query,dim],key[batch,query,key,dim],candidate_frames",
                "output_schema": "signed field[batch,query,candidate,key] and top2 candidate scores",
                "requires": ["frozen_score_witness", "candidate_frame_bank"],
                "conflicts": ["q_consensus_error_witness_classical_control"],
                "deterministic": True,
                "resource_estimate": {
                    "data_qubits": self.config.register_qubits,
                    "ancilla_qubits": 0,
                    "depth": self.config.depth,
                    "two_qubit_gates_per_state": self.config.depth * self.config.register_qubits,
                    "states_per_query_key": self.config.num_candidates,
                    "shots": "not modeled by exact statevector prototype",
                },
                "failure_signatures": [
                    "valid/test gain below the non-quantum gate",
                    "quantum field is query-independent",
                    "signed readout collapses under candidate permutation",
                    "matched classical product-state control wins",
                    "non-finite field or gradient",
                ],
            },
        }


class QuantumConsensusEstimator(ConsensusQuantumEstimator):
    kernel_type = "quantum"
    entangled_state = True


class ClassicalConsensusEstimator(ConsensusQuantumEstimator):
    kernel_type = "classical"
    entangled_state = False


def build_consensus_estimator(
    kernel_type: str,
    config: ConsensusQuantumEstimatorConfig,
    candidate_frames: torch.Tensor,
) -> ConsensusQuantumEstimator:
    if kernel_type == "quantum":
        return QuantumConsensusEstimator(config, candidate_frames)
    if kernel_type == "classical":
        return ClassicalConsensusEstimator(config, candidate_frames)
    raise ValueError(
        f"kernel_type must be one of {CONSENSUS_QUANTUM_KERNEL_TYPES}"
    )
