from __future__ import annotations

from q_attention.metrics import classification_metrics, classification_report


def test_classification_report_contains_micro_macro_and_per_class_metrics() -> None:
    report = classification_report(
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        3,
        label_names={0: "no_relation", 1: "located_in", 2: "founded_by"},
    )

    metrics = report["metrics"]
    assert metrics["accuracy"] == 0.75
    assert metrics["micro_precision"] == 0.75
    assert metrics["micro_recall"] == 0.75
    assert metrics["micro_f1"] == 0.75
    assert set(report["per_class"]) == {"0", "1", "2"}
    assert report["per_class"]["1"]["label"] == "located_in"
    assert report["per_class"]["1"]["support"] == 1
    assert report["per_class"]["2"]["false_negative"] == 1
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_classification_metrics_preserves_scalar_contract_and_adds_micro_fields() -> None:
    metrics = classification_metrics([0, 1], [0, 1], 2)

    assert set(metrics) == {
        "accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_classification_report_rejects_unknown_class_ids() -> None:
    try:
        classification_report([0, 2], [0, 1], 2)
    except ValueError as exc:
        assert "class IDs" in str(exc)
    else:
        raise AssertionError("expected invalid class IDs to be rejected")
