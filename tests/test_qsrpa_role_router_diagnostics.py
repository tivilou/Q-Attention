from __future__ import annotations

from experiments.diagnose_qsrpa_role_router_structure import run_diagnostics


def test_qsrpa_structural_diagnostic_is_label_free_and_deterministic() -> None:
    first = run_diagnostics()
    second = run_diagnostics()
    assert first == second
    assert first["trained_replay"] is False
    assert first["checkpoint_available"] is False
    assert first["label_free_action_path"] is True
    assert first["span_invariant_anchor"] is True
    assert first["query_independent_role_context"] is True
    assert first["query_features_change_when_query_changes"] is True
    assert first["role_weight_sum_max_error"] < 1e-6
    assert first["role_router_antisymmetry_max_error"] < 1e-6
