"""Torch-only quantum projector utilities.

The functions in this module are intentionally lightweight. They do not require
quantum simulators; instead, they provide an angle-encoding feature map and a
fidelity-style kernel that can build a key-space projector compatible with
``k' = k + gPk``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from q_attention.projectors import SpectralProjectorConfig, build_projector, spectral_filter_diagnostics

QUANTUM_KERNEL_MODES = ("fidelity", "centered_fidelity", "softmax_fidelity")


@dataclass(frozen=True)
class QuantumFeatureMapConfig:
    """Configuration for the quantum-inspired feature map."""

    num_qubits: int = 4
    angle_scale: float = 1.0
    seed: int = 17
    max_state_dim: int = 1024
    eps: float = 1e-8
    kernel_mode: str = "centered_fidelity"
    kernel_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.kernel_mode not in QUANTUM_KERNEL_MODES:
            raise ValueError(f"unknown quantum kernel mode: {self.kernel_mode}")
        if self.kernel_temperature <= 0:
            raise ValueError("kernel_temperature must be positive")


@dataclass(frozen=True)
class QuantumProjectorResult:
    """Projector result plus diagnostics useful for toy ablations."""

    projector: torch.Tensor
    features: torch.Tensor
    kernel: torch.Tensor
    singular_values: torch.Tensor
    metadata: dict[str, Any]


def _as_key_matrix(keys: torch.Tensor) -> torch.Tensor:
    if keys.ndim != 2:
        raise ValueError("keys must have shape (num_vectors, dim)")
    if keys.shape[0] == 0:
        raise ValueError("at least one key vector is required")
    return keys.float()


def deterministic_projection(input_dim: int, num_qubits: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic random projection from key space to qubit angles."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(input_dim, num_qubits, generator=generator, dtype=torch.float32)
    projection = projection / float(input_dim) ** 0.5
    return projection.to(device=device)


def angle_feature_map(keys: torch.Tensor, config: QuantumFeatureMapConfig | None = None) -> torch.Tensor:
    """Map key vectors to a tensor-product angle-encoded state."""
    config = config or QuantumFeatureMapConfig()
    state_dim = 2 ** config.num_qubits
    if state_dim > config.max_state_dim:
        raise ValueError(f"state dimension {state_dim} exceeds max_state_dim={config.max_state_dim}")

    x = _as_key_matrix(keys)
    x = F.normalize(x, p=2, dim=-1, eps=config.eps)
    projection = deterministic_projection(x.shape[-1], config.num_qubits, seed=config.seed, device=x.device)
    angles = config.angle_scale * torch.matmul(x, projection)

    state = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for qubit in range(config.num_qubits):
        local = torch.stack((torch.cos(angles[:, qubit]), torch.sin(angles[:, qubit])), dim=-1)
        state = (state.unsqueeze(-1) * local.unsqueeze(1)).reshape(x.shape[0], -1)
    return F.normalize(state, p=2, dim=-1, eps=config.eps)


def fidelity_kernel(features_a: torch.Tensor, features_b: torch.Tensor | None = None, *, eps: float = 1e-8) -> torch.Tensor:
    """Compute a fidelity-style kernel ``|<Phi(x), Phi(y)>|^2``."""
    if features_a.ndim != 2:
        raise ValueError("features_a must have shape (num_vectors, feature_dim)")
    features_b = features_a if features_b is None else features_b
    if features_b.ndim != 2 or features_b.shape[-1] != features_a.shape[-1]:
        raise ValueError("features_b must be 2D and share the feature dimension")
    a = features_a / features_a.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    b = features_b / features_b.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    overlap = torch.matmul(a, b.conj().transpose(0, 1))
    return overlap.abs().pow(2).clamp_min(0.0).float()


def transform_quantum_kernel(
    kernel: torch.Tensor,
    *,
    mode: str = "centered_fidelity",
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Transform a raw fidelity kernel before lifting it back into key space.

    ``centered_fidelity`` removes the near-constant component that made the first
    GPU run's quantum kernel almost uniform. ``softmax_fidelity`` keeps weights
    positive but sharpens row-wise neighborhood contrast.
    """
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D tensor")
    if mode not in QUANTUM_KERNEL_MODES:
        raise ValueError(f"unknown quantum kernel mode: {mode}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    k = kernel.float()
    if mode == "fidelity":
        return k

    if k.shape[0] != k.shape[1]:
        raise ValueError(f"{mode} requires a square self-kernel")

    if mode == "centered_fidelity":
        centered = k - k.mean(dim=0, keepdim=True) - k.mean(dim=1, keepdim=True) + k.mean()
        return 0.5 * (centered + centered.transpose(0, 1))

    scores = (k - k.mean(dim=-1, keepdim=True)) / temperature
    weights = torch.softmax(scores, dim=-1)
    symmetrized = 0.5 * (weights + weights.transpose(0, 1))
    return symmetrized.clamp_min(eps)


def quantum_weighted_covariance(
    keys: torch.Tensor,
    kernel: torch.Tensor,
    *,
    center: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Lift a quantum kernel back into key space as ``X.T @ K_q @ X``."""
    x = _as_key_matrix(keys)
    if kernel.shape != (x.shape[0], x.shape[0]):
        raise ValueError(f"kernel shape {kernel.shape} does not match num_vectors={x.shape[0]}")
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    k = kernel.to(device=x.device, dtype=x.dtype)
    normalizer = k.abs().sum().clamp_min(eps)
    omega = torch.matmul(x.transpose(0, 1), torch.matmul(k, x)) / normalizer
    return 0.5 * (omega + omega.transpose(0, 1))


def build_quantum_projector(
    keys: torch.Tensor,
    *,
    quantum_config: QuantumFeatureMapConfig | None = None,
    projector_config: SpectralProjectorConfig | None = None,
    center: bool = False,
) -> QuantumProjectorResult:
    """Build a key-space projector using a quantum-inspired kernel."""
    quantum_config = quantum_config or QuantumFeatureMapConfig()
    projector_config = projector_config or SpectralProjectorConfig()
    x = _as_key_matrix(keys)
    features = angle_feature_map(x, quantum_config)
    raw_kernel = fidelity_kernel(features, eps=quantum_config.eps)
    kernel = transform_quantum_kernel(
        raw_kernel,
        mode=quantum_config.kernel_mode,
        temperature=quantum_config.kernel_temperature,
        eps=quantum_config.eps,
    )
    omega = quantum_weighted_covariance(x, kernel, center=center, eps=quantum_config.eps)
    basis, singular_values, _ = torch.linalg.svd(omega, full_matrices=False)
    projector = build_projector(basis=basis, singular_values=singular_values, config=projector_config)
    metadata: dict[str, Any] = {
        "projector_family": "quantum_contrastive",
        "num_vectors": int(x.shape[0]),
        "key_dim": int(x.shape[1]),
        "state_dim": int(features.shape[1]),
        "center": center,
        "quantum_config": asdict(quantum_config),
        "projector_config": asdict(projector_config),
        "kernel_mode": quantum_config.kernel_mode,
        "kernel_trace": float(torch.trace(kernel).item()),
        "kernel_mean": float(kernel.mean().item()),
        "kernel_abs_mean": float(kernel.abs().mean().item()),
        "raw_kernel_trace": float(torch.trace(raw_kernel).item()),
        "raw_kernel_mean": float(raw_kernel.mean().item()),
        "top_singular_values": [float(value) for value in singular_values[: min(8, singular_values.numel())].tolist()],
        "filter_diagnostics": spectral_filter_diagnostics(singular_values, projector_config),
    }
    return QuantumProjectorResult(
        projector=projector,
        features=features,
        kernel=kernel,
        singular_values=singular_values,
        metadata=metadata,
    )


@dataclass(frozen=True)
class SupervisedQuantumProjectorConfig:
    """Configuration for the standalone label-aligned quantum projector."""

    num_qubits: int = 4
    depth: int = 2
    angle_scale: float = 1.0
    seed: int = 17
    max_state_dim: int = 1024
    max_train_samples: int = 256
    learning_rate: float = 0.05
    training_steps: int = 80
    kernel_mode: str = "centered_fidelity"
    kernel_temperature: float = 1.0
    exclude_alignment_diagonal: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if 2 ** self.num_qubits > self.max_state_dim:
            raise ValueError("quantum state dimension exceeds max_state_dim")
        if self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training_steps <= 0:
            raise ValueError("training_steps must be positive")
        if self.kernel_mode not in QUANTUM_KERNEL_MODES:
            raise ValueError(f"unknown quantum kernel mode: {self.kernel_mode}")
        if self.kernel_temperature <= 0:
            raise ValueError("kernel_temperature must be positive")


@dataclass(frozen=True)
class SupervisedQuantumProjectorResult:
    """Standalone supervised quantum projector and training diagnostics."""

    projector: torch.Tensor
    states: torch.Tensor
    kernel: torch.Tensor
    singular_values: torch.Tensor
    parameters: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def _as_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim)")
    if features.shape[0] == 0:
        raise ValueError("at least one feature vector is required")
    return features.float()


def _as_labels(labels: torch.Tensor, num_samples: int) -> torch.Tensor:
    if labels.ndim != 1 or labels.shape[0] != num_samples:
        raise ValueError("labels must have shape (num_samples,)")
    return labels.long()


def _balanced_indices(labels: torch.Tensor, max_samples: int, *, seed: int) -> torch.Tensor:
    """Select a deterministic approximately class-balanced subset."""
    labels = labels.detach().cpu().long()
    budget = min(int(max_samples), int(labels.shape[0]))
    if budget <= 0:
        raise ValueError("max_samples must leave at least one sample")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    classes = torch.unique(labels, sorted=True)
    per_class = {int(label): torch.where(labels == label)[0] for label in classes.tolist()}
    shuffled = {label: values[torch.randperm(values.numel(), generator=generator)] for label, values in per_class.items()}
    selected: list[int] = []
    cursors = {label: 0 for label in shuffled}
    while len(selected) < budget:
        progressed = False
        for label in shuffled:
            cursor = cursors[label]
            values = shuffled[label]
            if cursor >= values.numel():
                continue
            selected.append(int(values[cursor].item()))
            cursors[label] = cursor + 1
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
    return torch.tensor(selected, dtype=torch.long)


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    view = state.reshape(state.shape[0], 2**qubit, 2, 2 ** (num_qubits - qubit - 1))
    low = view[:, :, 0, :]
    high = view[:, :, 1, :]
    cosine = torch.cos(angles / 2).to(dtype=state.dtype).view(-1, 1, 1)
    sine = torch.sin(angles / 2).to(dtype=state.dtype).view(-1, 1, 1)
    out_low = cosine * low - sine * high
    out_high = sine * low + cosine * high
    return torch.stack((out_low, out_high), dim=2).reshape_as(state)


def _apply_rz(state: torch.Tensor, angles: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    view = state.reshape(state.shape[0], 2**qubit, 2, 2 ** (num_qubits - qubit - 1))
    phase = angles / 2
    low_phase = (torch.cos(phase) - 1j * torch.sin(phase)).to(dtype=state.dtype).view(-1, 1, 1)
    high_phase = (torch.cos(phase) + 1j * torch.sin(phase)).to(dtype=state.dtype).view(-1, 1, 1)
    out_low = view[:, :, 0, :] * low_phase
    out_high = view[:, :, 1, :] * high_phase
    return torch.stack((out_low, out_high), dim=2).reshape_as(state)


def _apply_cnot(state: torch.Tensor, control: int, target: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    control_mask = 1 << (num_qubits - control - 1)
    target_mask = 1 << (num_qubits - target - 1)
    permutation = torch.where((indices & control_mask) != 0, indices ^ target_mask, indices)
    return state[:, permutation]


def _initial_quantum_state(angles: torch.Tensor) -> torch.Tensor:
    state = torch.ones(angles.shape[0], 1, device=angles.device, dtype=torch.complex64)
    for qubit in range(angles.shape[1]):
        local = torch.stack((torch.cos(angles[:, qubit] / 2), torch.sin(angles[:, qubit] / 2)), dim=-1)
        state = (state.unsqueeze(-1) * local.to(dtype=state.dtype).unsqueeze(1)).reshape(angles.shape[0], -1)
    return state


def _initialize_quantum_parameters(config: SupervisedQuantumProjectorConfig, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    shape = (config.depth, config.num_qubits)
    parameters = {
        "ry_scale": torch.ones(shape, dtype=torch.float32),
        "ry_bias": 0.05 * torch.randn(shape, generator=generator, dtype=torch.float32),
        "rz_scale": 0.5 * torch.ones(shape, dtype=torch.float32),
        "rz_bias": 0.05 * torch.randn(shape, generator=generator, dtype=torch.float32),
    }
    return {name: value.to(device).detach().requires_grad_(True) for name, value in parameters.items()}


def parameterized_quantum_feature_map(
    features: torch.Tensor,
    config: SupervisedQuantumProjectorConfig,
    *,
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Encode features with trainable data re-uploading and ring-CNOT layers."""
    x = _as_feature_matrix(features)
    projection = deterministic_projection(x.shape[-1], config.num_qubits, seed=config.seed, device=x.device)
    x = F.normalize(x, p=2, dim=-1, eps=config.eps)
    angles = config.angle_scale * torch.matmul(x, projection)
    state = _initial_quantum_state(angles)
    for layer in range(config.depth):
        ry_angles = angles * parameters["ry_scale"][layer] + parameters["ry_bias"][layer]
        rz_angles = angles * parameters["rz_scale"][layer] + parameters["rz_bias"][layer]
        for qubit in range(config.num_qubits):
            state = _apply_ry(state, ry_angles[:, qubit], qubit, config.num_qubits)
            state = _apply_rz(state, rz_angles[:, qubit], qubit, config.num_qubits)
        if config.num_qubits > 1:
            for control in range(config.num_qubits):
                state = _apply_cnot(state, control, (control + 1) % config.num_qubits, config.num_qubits)
    return state / state.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(config.eps)


def kernel_target_alignment(
    kernel: torch.Tensor,
    labels: torch.Tensor,
    *,
    exclude_diagonal: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measure centered alignment between a quantum kernel and same-label targets."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    labels = _as_labels(labels, kernel.shape[0])
    target = labels.unsqueeze(0).eq(labels.unsqueeze(1)).to(dtype=kernel.dtype)
    aligned_kernel = kernel.real
    if exclude_diagonal:
        diagonal = torch.eye(kernel.shape[0], device=kernel.device, dtype=torch.bool)
        aligned_kernel = aligned_kernel.masked_fill(diagonal, 0.0)
        target = target.masked_fill(diagonal, 0.0)
    centering = lambda matrix: matrix - matrix.mean(dim=0, keepdim=True) - matrix.mean(dim=1, keepdim=True) + matrix.mean()
    centered_kernel = centering(aligned_kernel)
    centered_target = centering(target.real)
    numerator = (centered_kernel * centered_target).sum()
    denominator = centered_kernel.square().sum().sqrt() * centered_target.square().sum().sqrt()
    return numerator / denominator.clamp_min(eps)


def fit_supervised_quantum_feature_map(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: SupervisedQuantumProjectorConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fit circuit parameters using label-kernel alignment on a balanced subset."""
    x = _as_feature_matrix(features)
    y = _as_labels(labels, x.shape[0])
    indices = _balanced_indices(y, config.max_train_samples, seed=config.seed)
    device_indices = indices.to(x.device)
    train_x = x[device_indices]
    train_y = y.to(x.device)[device_indices]
    parameters = _initialize_quantum_parameters(config, x.device)
    optimizer = torch.optim.Adam(list(parameters.values()), lr=config.learning_rate)
    history: list[float] = []
    with torch.no_grad():
        initial_states = parameterized_quantum_feature_map(train_x, config, parameters=parameters)
        initial_alignment = float(
            kernel_target_alignment(
                fidelity_kernel(initial_states),
                train_y,
                exclude_diagonal=config.exclude_alignment_diagonal,
                eps=config.eps,
            ).item()
        )
    best_alignment = initial_alignment
    best_parameters = {name: value.detach().clone() for name, value in parameters.items()}
    for _ in range(config.training_steps):
        optimizer.zero_grad(set_to_none=True)
        states = parameterized_quantum_feature_map(train_x, config, parameters=parameters)
        alignment = kernel_target_alignment(
            fidelity_kernel(states),
            train_y,
            exclude_diagonal=config.exclude_alignment_diagonal,
            eps=config.eps,
        )
        current_alignment = float(alignment.detach().item())
        history.append(current_alignment)
        if current_alignment > best_alignment:
            best_alignment = current_alignment
            best_parameters = {name: value.detach().clone() for name, value in parameters.items()}
        loss = 1.0 - alignment
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameters["ry_scale"].clamp_(0.1, 3.0)
            parameters["rz_scale"].clamp_(0.1, 3.0)
            parameters["ry_bias"].clamp_(-math.pi, math.pi)
            parameters["rz_bias"].clamp_(-math.pi, math.pi)
    with torch.no_grad():
        final_states = parameterized_quantum_feature_map(train_x, config, parameters=parameters)
        last_alignment = float(
            kernel_target_alignment(
                fidelity_kernel(final_states),
                train_y,
                exclude_diagonal=config.exclude_alignment_diagonal,
                eps=config.eps,
            ).item()
        )
    history.append(last_alignment)
    if last_alignment > best_alignment:
        best_alignment = last_alignment
        best_parameters = {name: value.detach().clone() for name, value in parameters.items()}
    diagnostics = {
        "num_train_samples": int(indices.numel()),
        "initial_alignment": initial_alignment,
        "final_alignment": best_alignment,
        "last_alignment": last_alignment,
        "alignment_history_tail": history[-5:],
        "num_labels": int(torch.unique(y).numel()),
        "label_counts": {str(int(label)): int((y == label).sum().item()) for label in torch.unique(y, sorted=True)},
    }
    return best_parameters, diagnostics


def build_supervised_quantum_projector(
    keys: torch.Tensor,
    relation_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    quantum_config: SupervisedQuantumProjectorConfig | None = None,
    projector_config: SpectralProjectorConfig | None = None,
    center: bool = True,
    max_vectors: int | None = None,
) -> SupervisedQuantumProjectorResult:
    """Build a key-space projector from a label-aligned quantum kernel alone."""
    quantum_config = quantum_config or SupervisedQuantumProjectorConfig()
    projector_config = projector_config or SpectralProjectorConfig()
    x = _as_key_matrix(keys)
    z = _as_feature_matrix(relation_features).to(x.device)
    y = _as_labels(labels, x.shape[0]).to(x.device)
    if z.shape[0] != x.shape[0]:
        raise ValueError("keys and relation_features must contain the same number of samples")
    parameters, training_diagnostics = fit_supervised_quantum_feature_map(z, y, quantum_config)
    projection_indices = _balanced_indices(y, max_vectors or x.shape[0], seed=quantum_config.seed + 1).to(x.device)
    projection_x = x[projection_indices]
    projection_z = z[projection_indices]
    projection_y = y[projection_indices]
    states = parameterized_quantum_feature_map(projection_z, quantum_config, parameters=parameters)
    raw_kernel = fidelity_kernel(states, eps=quantum_config.eps)
    kernel = transform_quantum_kernel(
        raw_kernel,
        mode=quantum_config.kernel_mode,
        temperature=quantum_config.kernel_temperature,
        eps=quantum_config.eps,
    )
    omega = quantum_weighted_covariance(projection_x, kernel, center=center, eps=quantum_config.eps)
    basis, singular_values, _ = torch.linalg.svd(omega, full_matrices=False)
    projector = build_projector(basis=basis, singular_values=singular_values, config=projector_config)
    metadata: dict[str, Any] = {
        "projector_family": "quantum_label_aligned",
        "standalone": True,
        "num_vectors": int(projection_x.shape[0]),
        "key_dim": int(projection_x.shape[1]),
        "relation_feature_dim": int(projection_z.shape[1]),
        "state_dim": int(states.shape[1]),
        "center": center,
        "quantum_config": asdict(quantum_config),
        "projector_config": asdict(projector_config),
        "kernel_mode": quantum_config.kernel_mode,
        "kernel_trace": float(torch.trace(kernel).item()),
        "kernel_mean": float(kernel.mean().item()),
        "raw_kernel_mean": float(raw_kernel.mean().item()),
        "label_counts": {str(int(label)): int((projection_y == label).sum().item()) for label in torch.unique(projection_y, sorted=True)},
        "training": training_diagnostics,
        "circuit": {
            "depth": quantum_config.depth,
            "entanglement": "ring_cnot",
            "trainable_parameter_count": sum(int(value.numel()) for value in parameters.values()),
        },
        "top_singular_values": [float(value) for value in singular_values[: min(8, singular_values.numel())].tolist()],
        "filter_diagnostics": spectral_filter_diagnostics(singular_values, projector_config),
    }
    return SupervisedQuantumProjectorResult(
        projector=projector,
        states=states,
        kernel=kernel,
        singular_values=singular_values,
        parameters={name: value.detach().cpu() for name, value in parameters.items()},
        metadata=metadata,
    )


def build_quantum_residual_projector(
    classical_projector: torch.Tensor,
    quantum_projector: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Build the optional classical-plus-quantum residual ablation."""
    if classical_projector.shape != quantum_projector.shape or classical_projector.ndim != 2:
        raise ValueError("classical and quantum projectors must have the same square shape")
    if classical_projector.shape[0] != classical_projector.shape[1]:
        raise ValueError("projectors must be square")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    quantum_projector = quantum_projector.to(device=classical_projector.device, dtype=classical_projector.dtype)
    identity = torch.eye(classical_projector.shape[0], device=classical_projector.device, dtype=classical_projector.dtype)
    complement = identity - classical_projector
    residual = complement @ quantum_projector @ complement
    return 0.5 * (classical_projector + alpha * residual + (classical_projector + alpha * residual).transpose(0, 1))
