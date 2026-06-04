"""Small classification metrics used by experiment scripts."""

from __future__ import annotations


def classification_metrics(predictions: list[int], labels: list[int], num_labels: int) -> dict[str, float]:
    """Compute accuracy and macro precision/recall/F1 without external deps."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        raise ValueError("at least one label is required")

    correct = sum(int(pred == gold) for pred, gold in zip(predictions, labels))
    accuracy = correct / len(labels)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label_id in range(num_labels):
        tp = sum(int(pred == label_id and gold == label_id) for pred, gold in zip(predictions, labels))
        fp = sum(int(pred == label_id and gold != label_id) for pred, gold in zip(predictions, labels))
        fn = sum(int(pred != label_id and gold == label_id) for pred, gold in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / num_labels,
        "macro_recall": sum(recalls) / num_labels,
        "macro_f1": sum(f1s) / num_labels,
    }
