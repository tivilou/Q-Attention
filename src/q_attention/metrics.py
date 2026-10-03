"""Classification metrics shared by experiment and report pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _validate_inputs(predictions: Sequence[int], labels: Sequence[int], num_labels: int) -> None:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        raise ValueError("at least one label is required")
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")
    invalid = [
        value
        for value in (*predictions, *labels)
        if not isinstance(value, int) or value < 0 or value >= num_labels
    ]
    if invalid:
        raise ValueError(f"class IDs must be in [0, {num_labels}); got {invalid[:4]}")


def classification_report(
    predictions: Sequence[int],
    labels: Sequence[int],
    num_labels: int,
    *,
    label_names: Mapping[int, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return complete single-label classification evidence.

    The summary keeps the historical scalar fields and adds micro metrics. The
    detailed payload is kept beside ``metrics`` so existing callers that iterate
    over scalar metrics do not accidentally treat a nested per-class table as a
    number.
    """
    _validate_inputs(predictions, labels, num_labels)

    def label_name(label_id: int) -> str:
        if label_names is None:
            return str(label_id)
        if isinstance(label_names, Mapping):
            return str(label_names.get(label_id, label_id))
        if label_id < len(label_names):
            return str(label_names[label_id])
        return str(label_id)

    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    for prediction, gold in zip(predictions, labels):
        confusion[gold][prediction] += 1

    for label_id in range(num_labels):
        true_positive = confusion[label_id][label_id]
        false_positive = sum(confusion[row][label_id] for row in range(num_labels) if row != label_id)
        false_negative = sum(confusion[label_id][column] for column in range(num_labels) if column != label_id)
        support = sum(confusion[label_id])
        predicted_support = sum(confusion[row][label_id] for row in range(num_labels))
        precision = true_positive / predicted_support if predicted_support else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative
        per_class[str(label_id)] = {
            "label_id": label_id,
            "label": label_name(label_id),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_support": predicted_support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    total = len(labels)
    accuracy = total_true_positive / total
    micro_precision = (
        total_true_positive / (total_true_positive + total_false_positive)
        if total_true_positive + total_false_positive
        else 0.0
    )
    micro_recall = (
        total_true_positive / (total_true_positive + total_false_negative)
        if total_true_positive + total_false_negative
        else 0.0
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "metrics": {
            "accuracy": accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": sum(precisions) / num_labels,
            "macro_recall": sum(recalls) / num_labels,
            "macro_f1": sum(f1s) / num_labels,
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
        "num_items": total,
        "num_labels": num_labels,
    }


def classification_metrics(predictions: list[int], labels: list[int], num_labels: int) -> dict[str, float]:
    """Compute scalar accuracy, micro and macro metrics without external deps."""
    report = classification_report(predictions, labels, num_labels)
    return dict(report["metrics"])


def correct_label_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the gold logit minus the strongest competing logit per example."""
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], torch.finfo(logits.dtype).min)
    return correct - competitors.max(dim=-1).values
