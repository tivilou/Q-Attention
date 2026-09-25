#!/usr/bin/env python3
"""Fixed, analytic coherent-path motif audit for Q-WAP Stage 0."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import torch


WALK_TIME = float(torch.pi / 4)
ANCHOR = 0
TARGET = 3
DISTRACTOR = 4
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4))
EDGE_SIGNS = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0)


def signed_motif() -> torch.Tensor:
    graph = torch.zeros(5, 5, dtype=torch.float64)
    for (source, target), sign in zip(EDGES, EDGE_SIGNS):
        graph[source, target] = sign
        graph[target, source] = sign
    return graph


def walk_probabilities(graph: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(graph)
    complex_vectors = eigenvectors.to(torch.complex128)
    phases = torch.exp((-1j * WALK_TIME * eigenvalues).to(torch.complex128))
    unitary = (complex_vectors * phases) @ complex_vectors.T.conj()
    return unitary.abs().square()[ANCHOR].real


def classical_two_step_probabilities(graph: torch.Tensor) -> torch.Tensor:
    adjacency = graph.abs()
    transition = adjacency / adjacency.sum(dim=-1, keepdim=True)
    return (transition @ transition)[ANCHOR]


def audit() -> dict[str, Any]:
    graph = signed_motif()
    quantum = walk_probabilities(graph)
    classical = classical_two_step_probabilities(graph)
    quantum_gap = float(quantum[TARGET] - quantum[DISTRACTOR])
    classical_gap = float(classical[TARGET] - classical[DISTRACTOR])
    conditions = {
        "fixed_walk_time": abs(WALK_TIME - float(torch.pi / 4)) <= 1e-15,
        "classical_target_distractor_tie": abs(classical_gap) <= 1e-12,
        "quantum_target_distractor_separation": quantum_gap >= 0.20,
        "finite_probabilities": bool(torch.isfinite(quantum).all()),
        "probability_normalization": abs(float(quantum.sum()) - 1.0) <= 1e-10,
    }
    passed = all(conditions.values())
    return {
        "schema_version": "q-attention.coherent-path-motif-audit.v1",
        "status": "pass" if passed else "fail",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "graph": {
            "nodes": 5,
            "anchor": ANCHOR,
            "target": TARGET,
            "distractor": DISTRACTOR,
            "edges": [list(edge) for edge in EDGES],
            "edge_signs": list(EDGE_SIGNS),
        },
        "walk_time": WALK_TIME,
        "quantum_probabilities": quantum.tolist(),
        "classical_two_step_probabilities": classical.tolist(),
        "target_minus_distractor": {
            "quantum": quantum_gap,
            "classical": classical_gap,
        },
        "conditions": conditions,
        "interpretation": {
            "supported": "Signed path interference can separate target from distractor when nonnegative two-step diffusion is tied.",
            "not_supported": "This motif does not establish attention task utility, label-free selection, or quantum hardware advantage.",
            "next": "Use the same fixed walk time and signed path construction in a query/entity-anchored score-hook canary; do not tune the motif after seeing task results.",
        },
    }


def main() -> None:
    output_root = Path("runs/q_coherent_path_motif_audit")
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = audit()
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "status": summary["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
