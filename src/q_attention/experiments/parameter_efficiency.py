"""Shared parameter and ideal-circuit resource accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


def trainable_parameter_count(module: object) -> int:
    """Count all parameters exposed by a module that require gradients."""
    named_parameters = getattr(module, "named_parameters", None)
    if named_parameters is None:
        raise TypeError("module must expose named_parameters()")
    return int(
        sum(
            parameter.numel()
            for _name, parameter in named_parameters()
            if bool(parameter.requires_grad)
        )
    )


def trainable_parameter_breakdown(module: object) -> dict[str, int]:
    """Return a stable per-tensor trainable-parameter breakdown."""
    named_parameters = getattr(module, "named_parameters", None)
    if named_parameters is None:
        raise TypeError("module must expose named_parameters()")
    return {
        str(name): int(parameter.numel())
        for name, parameter in named_parameters()
        if bool(parameter.requires_grad)
    }


@dataclass(frozen=True)
class QuantumResourceLedger:
    """Resource fields for a conditional ideal-quantum claim."""

    logical_qubits: int | None = None
    circuit_depth: int | None = None
    one_qubit_gates: int | None = None
    two_qubit_gates: int | None = None
    circuit_evaluations: int | None = None
    shots_per_evaluation: int | None = None
    state_preparation_gates: int | None = None
    measurement_observables: int | None = None
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_qubits": self.logical_qubits,
            "circuit_depth": self.circuit_depth,
            "one_qubit_gates": self.one_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "circuit_evaluations": self.circuit_evaluations,
            "shots_per_evaluation": self.shots_per_evaluation,
            "state_preparation_gates": self.state_preparation_gates,
            "measurement_observables": self.measurement_observables,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class ParameterEfficiencyManifest:
    """Complete accounting record for one candidate/control configuration."""

    candidate_id: str
    mechanism: str
    components: Mapping[str, int]
    resources: QuantumResourceLedger
    controls: tuple[str, ...] = ()
    code_revision: str = "unknown"
    dataset_identity: str = "not_run"
    claims: tuple[str, ...] = (
        "task_utility",
        "trainable_parameter_efficiency",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_trainable_parameters(self) -> int:
        return int(sum(int(value) for value in self.components.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "q-attention.parameter-efficiency.v1",
            "candidate_id": self.candidate_id,
            "mechanism": self.mechanism,
            "components": {str(key): int(value) for key, value in self.components.items()},
            "total_trainable_parameters": self.total_trainable_parameters,
            "resources": self.resources.to_dict(),
            "controls": list(self.controls),
            "code_revision": self.code_revision,
            "dataset_identity": self.dataset_identity,
            "claims": list(self.claims),
            "metadata": dict(self.metadata),
        }


def build_parameter_efficiency_manifest(
    *,
    candidate_id: str,
    mechanism: str,
    modules: Mapping[str, object],
    resources: QuantumResourceLedger,
    controls: tuple[str, ...] = (),
    code_revision: str = "unknown",
    dataset_identity: str = "not_run",
    claims: tuple[str, ...] = (
        "task_utility",
        "trainable_parameter_efficiency",
    ),
    metadata: Mapping[str, Any] | None = None,
) -> ParameterEfficiencyManifest:
    """Build a manifest while counting every supplied trainable component."""
    components = {
        str(name): trainable_parameter_count(module)
        for name, module in modules.items()
    }
    return ParameterEfficiencyManifest(
        candidate_id=candidate_id,
        mechanism=mechanism,
        components=components,
        resources=resources,
        controls=controls,
        code_revision=code_revision,
        dataset_identity=dataset_identity,
        claims=claims,
        metadata={} if metadata is None else dict(metadata),
    )


def write_parameter_efficiency_manifest(
    manifest: ParameterEfficiencyManifest,
    path: str | Path,
) -> None:
    """Write a deterministic JSON manifest and create its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
