from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_registry_freezes_reported_top_five_without_claiming_formal_results() -> None:
    payload = json.loads(
        (ROOT / "configs" / "retacred_reference_baselines.json").read_text(encoding="utf-8")
    )

    assert payload["schema"] == "q-attention.retacred-reference-baseline-registry.v2"
    assert payload["dataset"] == "Re-TACRED"
    assert payload["selection"]["metric"] == "micro_f1"
    assert payload["selection"]["scores_are_formal_project_results"] is False

    candidates = payload["candidates"]
    assert [candidate["rank"] for candidate in candidates] == [1, 2, 3, 4, 5]
    assert len({candidate["id"] for candidate in candidates}) == 5
    assert {candidate["method_family"] for candidate in candidates} >= {
        "sequence_routing",
        "ctl_drp",
        "curriculum_learning",
        "typed_entity_marker",
    }

    for candidate in candidates:
        assert candidate["status"] in {"planned", "blocked_source_and_protocol"}
        assert candidate["strict_comparison_eligible"] is False
        assert 0.0 < candidate["reported_in_paper"]["test_micro_f1"] < 1.0
        assert candidate["paper_url"].startswith("https://")
        assert candidate["required_gates"]

    assert payload["excluded_or_non_strict"]
    assert payload["interpretation"]["reported_scores_are_not_project_measurements"] is True

    routing = next(row for row in candidates if row["method_family"] == "sequence_routing")
    assert routing["status"] == "blocked_source_and_protocol"
    assert routing["source_commit"] == "cc95c4ad20b9ed46b978ffa4e88e31ce28c7ed20"
    assert routing["reported_in_paper"]["test_micro_f1"] == .922
    assert routing["reported_score_rank_eligible"] is False
    assert routing["reported_in_paper"]["evaluator_binding"] == "unverified"
    assert "including_no_relation" in routing["released_source_metric"]


def test_first_reproduction_manifest_keeps_source_and_adaptation_gates_separate() -> None:
    manifest = json.loads(
        (
            ROOT
            / "configs"
            / "reference_baselines"
            / "improved_entity_marker_roberta_large.manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["provenance"]["mode"] == "source"
    assert manifest["provenance"]["commit"] == "fe00781116ab4c34cbcd9a2ae5b02513afae87ee"
    assert manifest["provenance"]["checkpoint"] is None
    assert manifest["fidelity_gates"]["status"] == "planned"
    assert manifest["comparison"]["strict_comparison_eligible"] is False
    assert manifest["adaptation_boundary"]["allowed"]
    assert manifest["adaptation_boundary"]["forbidden"]
