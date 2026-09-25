#!/usr/bin/env python3
"""Audit ideal-circuit resources for the frozen Q-LASS and Q-VRES designs.

This is a read-only accounting pass. It performs no training, data loading, or
quantum simulation. Counts are transparent lower bounds under explicitly
declared measurement assumptions; they are not hardware runtime claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResourceModel:
    circuit_id: str
    qubits: int
    depth: int
    trainable_parameters: int
    classical_control_parameters: int
    encoded_states_per_query_key: int | str
    keys_per_query: int | str
    two_qubit_gates_per_state: int
    one_qubit_rotations_per_state: int
    state_preparation_calls_per_query_key: int | str
    state_preparation_calls_per_query: int | str
    oracle_queries_per_query: int | str
    readout_observables_per_query_key: int | str
    readout_observables_per_query: int | str
    shots: int | None
    ancilla_qubits: int
    overlap_readout_overhead: str
    measurement_assumption: str
    precision_target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/q_ideal_quantum_resource_audit.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_commit": result.stdout.strip() if result.returncode == 0 else "unknown",
        "git_dirty": bool(status.stdout.strip()),
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.ideal-quantum-resource-audit.v1":
        raise ValueError("unsupported ideal quantum resource audit config")
    for name in ("qlass", "qvres"):
        if name not in payload:
            raise ValueError(f"missing frozen design: {name}")
    return payload


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def qlass_model(cfg: dict[str, Any]) -> ResourceModel:
    q = _positive_int(cfg["qubits"], "Q-LASS qubits")
    depth = _positive_int(cfg["depth"], "Q-LASS depth")
    candidates = _positive_int(cfg["num_candidates"], "Q-LASS candidates")
    head_dim = _positive_int(cfg["head_dim"], "Q-LASS head_dim")
    keys = _positive_int(cfg["keys_per_query"], "Q-LASS keys_per_query")
    # Projection (3 * head_dim) x qubits, plus depth x qubits scales and biases.
    params = 3 * head_dim * q + 2 * depth * q
    # The product control has the same trainable tensors by construction.
    states = candidates
    # Each data-reuploading layer applies one ring CNOT per qubit.
    two_qubit = depth * q
    return ResourceModel(
        circuit_id="q-lass",
        qubits=q,
        depth=depth,
        trainable_parameters=params,
        classical_control_parameters=params,
        encoded_states_per_query_key=states,
        keys_per_query=keys,
        two_qubit_gates_per_state=two_qubit,
        one_qubit_rotations_per_state=(depth + 1) * q,
        state_preparation_calls_per_query_key=states,
        state_preparation_calls_per_query=states * keys,
        oracle_queries_per_query="not specified",
        readout_observables_per_query_key=states,
        readout_observables_per_query=states * keys,
        shots=None,
        ancilla_qubits=0,
        overlap_readout_overhead="none; direct parity measurement assumption",
        measurement_assumption="one parity expectation per candidate state",
        precision_target=str(cfg.get("precision_target", "not_specified")),
    )


def qvres_model(cfg: dict[str, Any]) -> ResourceModel:
    q = _positive_int(cfg["qubits"], "Q-VRES qubits")
    depth = _positive_int(cfg["depth"], "Q-VRES depth")
    layers = _positive_int(cfg["num_layers"], "Q-VRES layers")
    heads = _positive_int(cfg["num_heads"], "Q-VRES heads")
    head_dim = _positive_int(cfg["head_dim"], "Q-VRES head_dim")
    keys = cfg["keys_per_query"]
    # query/token scales and biases: 4 * layers * heads * depth * q;
    # raw transport: layers * heads.
    params = 4 * layers * heads * depth * q + layers * heads
    return ResourceModel(
        circuit_id="q-vres",
        qubits=q,
        depth=depth,
        trainable_parameters=params,
        classical_control_parameters=params,
        encoded_states_per_query_key=2,
        keys_per_query=keys,
        two_qubit_gates_per_state=depth * q,
        one_qubit_rotations_per_state=(depth + 1) * q,
        state_preparation_calls_per_query_key=2,
        state_preparation_calls_per_query="num_layers * num_heads * 2 * keys_per_query",
        oracle_queries_per_query="not specified",
        readout_observables_per_query_key="num_layers * num_heads",
        readout_observables_per_query="num_layers * num_heads * keys_per_query",
        shots=None,
        ancilla_qubits=1,
        overlap_readout_overhead=(
            "one SWAP-test ancilla and one controlled-SWAP per register qubit per overlap; "
            "gate decomposition not specified"
        ),
        measurement_assumption="one overlap/fidelity estimate per query-token pair",
        precision_target=str(cfg.get("precision_target", "not_specified")),
    )


def classification(model: ResourceModel, *, utility_status: str, attribution_status: str) -> dict[str, Any]:
    parameter_advantage = model.trainable_parameters < model.classical_control_parameters
    ideal_resource_advantage = False
    if parameter_advantage:
        resource_status = "parameter_advantage"
    elif attribution_status == "pass" and ideal_resource_advantage:
        resource_status = "ideal_circuit_resource_advantage"
    else:
        resource_status = "resource_advantage_not_established"
    return {
        "parameter_advantage": parameter_advantage,
        "ideal_circuit_resource_advantage": ideal_resource_advantage,
        "resource_advantage_status": resource_status,
        "ordinary_method_utility_status": utility_status,
        "quantum_attribution_status": attribution_status,
        "claim_ceiling": (
            "ordinary_method_utility_only"
            if resource_status == "resource_advantage_not_established"
            and utility_status in {"reproducible_synthetic_utility", "positive_task_utility"}
            else "no_positive_utility_or_resource_claim"
            if resource_status == "resource_advantage_not_established"
            else resource_status
        ),
    }


def audit(config: dict[str, Any]) -> dict[str, Any]:
    qlass = qlass_model(config["qlass"])
    qvres = qvres_model(config["qvres"])
    models = {"qlass": qlass, "qvres": qvres}
    result: dict[str, Any] = {
        "schema_version": "q-attention.ideal-quantum-resource-audit.v1",
        "status": "complete",
        "audit_type": "read_only_symbolic_resource_accounting",
        "environment": {"python": platform.python_version()},
        "assumptions": {
            "state_preparation_included": True,
            "classical_angle_encoding_cost": "not quantified",
            "oracle_and_data_loading": "not specified by either implementation",
            "measurement_shots": "not modeled; no precision target supplied",
            "classical_optimization": "not included in circuit resource counts",
            "hardware_runtime_or_energy": "not inferred",
        },
        "models": {},
        "cross_model_conclusions": {
            "parameter_advantage": "not established: quantum and matched classical controls use equal trainable parameter counts",
            "ideal_circuit_resource_advantage": "not established: both designs require repeated state preparation and measurement, while oracle, loading, shots, and precision are unspecified",
            "current_hardware_advantage": "not assessed",
            "ordinary_task_utility": "separate axis; Q-LASS has reproducible synthetic utility, Q-VRES formal Re-TACRED selector result is below baseline and classical control",
        },
    }
    for name, model in models.items():
        utility = "reproducible_synthetic_utility" if name == "qlass" else "formal_natural_task_negative"
        attribution = "blocked_by_matched_control" if name == "qlass" else "blocked_by_matched_control_and_negative_gate"
        item = asdict(model)
        item.update(
            {
                key: config[name][key]
                for key in (
                    "id",
                    "source",
                    "control",
                    "readout",
                    "measurement",
                    "state_preparation",
                    "data_loading",
                    "oracle_query",
                    "classical_optimization",
                    "evidence",
                )
                if key in config[name]
            }
        )
        item["gate_counts"] = {
            "two_qubit_gates_per_query_key_lower_bound": (
                model.two_qubit_gates_per_state * int(model.encoded_states_per_query_key)
                if isinstance(model.encoded_states_per_query_key, int)
                else "depth * qubits * 2"
            ),
            "two_qubit_gates_per_query_lower_bound": (
                model.two_qubit_gates_per_state
                * int(model.state_preparation_calls_per_query)
                if isinstance(model.state_preparation_calls_per_query, int)
                else "num_layers * num_heads * 2 * depth * qubits * keys_per_query"
            ),
            "one_qubit_rotations_per_query_lower_bound": (
                model.one_qubit_rotations_per_state
                * int(model.state_preparation_calls_per_query)
                if isinstance(model.state_preparation_calls_per_query, int)
                else "num_layers * num_heads * 2 * (depth + 1) * qubits * keys_per_query"
            ),
            "readout_observables_per_query_key": model.readout_observables_per_query_key,
            "readout_observables_per_query": model.readout_observables_per_query,
        }
        item["classification"] = classification(
            model, utility_status=utility, attribution_status=attribution
        )
        result["models"][name] = item
    return result


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    payload = audit(config)
    payload["config_path"] = config_path.relative_to(ROOT).as_posix()
    payload["config_sha256"] = sha256(config_path)
    payload["provenance"] = git_revision()
    output = resolve(args.output) if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
