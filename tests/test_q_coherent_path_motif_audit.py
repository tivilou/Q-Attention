from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_path_motif_audit import audit  # noqa: E402


def test_fixed_motif_has_quantum_only_target_separation() -> None:
    summary = audit()
    assert summary["status"] == "pass"
    assert summary["target_minus_distractor"]["classical"] == 0.0
    assert summary["target_minus_distractor"]["quantum"] >= 0.20
    assert all(summary["conditions"].values())
