from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_causal_value_evidence_relation_transfer import (  # noqa: E402
    SELECTORS,
    _masked_attention,
    build_kernel,
    git_output,
    hook_config,
)
from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.adapters.encoder import resolve_module  # noqa: E402
from q_attention.experiments.progress import tracked_batches  # noqa: E402
from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
    move_batch,
    torch_load_weights,
)
from q_attention.metrics import classification_metrics  # noqa: E402
from q_attention.tasks.relation import (  # noqa: E402
    RelationRecord,
    load_relation_jsonl,
    sample_relation_records_proportional,
)


DEFAULT_ATTRIBUTION_SELECTORS = (
    "q_causal_transport",
    "q_causal_key_only",
)


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    baseline_dir: Path
    data_path: Path
    selector_dirs: Mapping[str, Path]
    selector_configs: Mapping[str, Path]
    parallel_mode: str


@dataclass
class ScalarSummary:
    count: int = 0
    total: float = 0.0
    square_total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.square_total += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class MechanismAccumulator:
    layer_metrics: list[defaultdict[str, ScalarSummary]]
    relation_metrics: defaultdict[str, defaultdict[str, ScalarSummary]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(ScalarSummary))
    )

    @classmethod
    def create(cls, num_layers: int) -> "MechanismAccumulator":
        return cls([defaultdict(ScalarSummary) for _ in range(num_layers)])

    def add(self, layer_index: int, relation: str, metrics: Mapping[str, float | None]) -> None:
        for name, value in metrics.items():
            self.layer_metrics[layer_index][name].add(value)
            self.relation_metrics[relation][name].add(value)

    def result(self) -> dict[str, Any]:
        return {
            "layers": [
                {
                    "layer_index": layer_index,
                    "metrics": {
                        name: summary.result()
                        for name, summary in sorted(metrics.items())
                    },
                }
                for layer_index, metrics in enumerate(self.layer_metrics)
            ],
            "per_relation": {
                relation: {
                    name: summary.result()
                    for name, summary in sorted(metrics.items())
                }
                for relation, metrics in sorted(self.relation_metrics.items())
            },
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def resolve_run_layout(run_dir: str | Path, split: str) -> RunLayout:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    parallel_baseline = run_dir / "stages" / "baseline" / "baseline"
    if parallel_baseline.is_dir():
        baseline_stage = run_dir / "stages" / "baseline"
        selector_dirs = {
            selector: run_dir / "stages" / selector / "selectors" / selector
            for selector in SELECTORS
            if selector != "disabled"
        }
        selector_configs = {
            selector: run_dir / "stages" / selector / "run_config.json"
            for selector in SELECTORS
            if selector != "disabled"
        }
        layout = RunLayout(
            run_dir=run_dir,
            baseline_dir=parallel_baseline,
            data_path=baseline_stage / "screen_data" / f"{split}.jsonl",
            selector_dirs=selector_dirs,
            selector_configs=selector_configs,
            parallel_mode="selectors",
        )
    else:
        layout = RunLayout(
            run_dir=run_dir,
            baseline_dir=run_dir / "baseline",
            data_path=run_dir / "screen_data" / f"{split}.jsonl",
            selector_dirs={
                selector: run_dir / "selectors" / selector
                for selector in SELECTORS
                if selector != "disabled"
            },
            selector_configs={
                selector: run_dir / "run_config.json"
                for selector in SELECTORS
                if selector != "disabled"
            },
            parallel_mode="serial",
        )
    required = [
        layout.baseline_dir / "model.pt",
        layout.baseline_dir / "metrics.json",
        layout.data_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("raw run is incomplete; missing: " + ", ".join(missing))
    return layout


def _selector_args(config_path: Path) -> SimpleNamespace:
    config = _read_json(config_path)
    defaults = {
        "seed": 13,
        "register_qubits": 2,
        "depth": 1,
        "angle_scale": 1.0,
        "max_transport": 0.25,
        "initial_transport": 0.05,
        "evidence_floor": 1e-6,
    }
    defaults.update({key: config[key] for key in defaults if key in config})
    return SimpleNamespace(**defaults)


def load_selector_kernel(
    selector: str,
    layout: RunLayout,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    selector_dir = layout.selector_dirs[selector]
    checkpoint = selector_dir / "best_kernel.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing selector checkpoint: {checkpoint}")
    args = _selector_args(layout.selector_configs[selector])
    kernel = build_kernel(selector, model, int(args.seed), args)
    if kernel is None:
        raise ValueError(f"selector {selector} does not have a kernel")
    kernel.load_state_dict(torch_load_weights(checkpoint, map_location="cpu"))
    kernel.to(device)
    kernel.eval()
    return kernel


def _gold_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gold = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    return gold - competitors.max(dim=1).values


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    centered_a = [value - mean_a for value in ranks_a]
    centered_b = [value - mean_b for value in ranks_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def topk_overlap(values_a: Sequence[float], values_b: Sequence[float], fraction: float = 0.2) -> float | None:
    if len(values_a) != len(values_b) or not values_a:
        return None
    k = max(1, math.ceil(len(values_a) * fraction))
    top_a = set(sorted(range(len(values_a)), key=lambda index: values_a[index], reverse=True)[:k])
    top_b = set(sorted(range(len(values_b)), key=lambda index: values_b[index], reverse=True)[:k])
    return len(top_a & top_b) / k


def relation_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    id_to_label: Mapping[int, str],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label_id, label_name in sorted(id_to_label.items(), key=lambda item: item[1]):
        support = sum(gold == label_id for gold in labels)
        predicted = sum(value == label_id for value in predictions)
        true_positive = sum(
            gold == label_id and prediction == label_id
            for prediction, gold in zip(predictions, labels)
        )
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        result[label_name] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def add_relation_deltas(
    current: Mapping[str, Mapping[str, float | int]],
    baseline: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for relation, metrics in current.items():
        row = dict(metrics)
        for name in ("precision", "recall", "f1"):
            row[f"delta_{name}"] = float(metrics[name]) - float(baseline[relation][name])
        result[relation] = row
    return result


def _support_quartiles(
    per_relation: Mapping[str, Mapping[str, float | int]],
) -> list[tuple[str, list[str]]]:
    relations = [
        relation
        for relation, metrics in sorted(
            per_relation.items(), key=lambda item: (int(item[1]["support"]), item[0])
        )
        if int(metrics["support"]) > 0
    ]
    names = ("q1_rarest", "q2", "q3", "q4_most_frequent")
    bins: list[tuple[str, list[str]]] = []
    for index, name in enumerate(names):
        start = len(relations) * index // 4
        end = len(relations) * (index + 1) // 4
        bins.append((name, relations[start:end]))
    return bins


def support_analysis(
    per_relation: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    for name, relations in _support_quartiles(per_relation):
        supports = [int(per_relation[relation]["support"]) for relation in relations]
        recall_deltas = [float(per_relation[relation]["delta_recall"]) for relation in relations]
        f1_deltas = [float(per_relation[relation]["delta_f1"]) for relation in relations]
        total_support = sum(supports)
        bins.append(
            {
                "name": name,
                "relation_count": len(relations),
                "support_min": min(supports) if supports else None,
                "support_max": max(supports) if supports else None,
                "mean_recall_delta": sum(recall_deltas) / len(recall_deltas) if recall_deltas else None,
                "mean_f1_delta": sum(f1_deltas) / len(f1_deltas) if f1_deltas else None,
                "support_weighted_recall_delta": (
                    sum(support * delta for support, delta in zip(supports, recall_deltas)) / total_support
                    if total_support
                    else None
                ),
            }
        )
    active = [metrics for metrics in per_relation.values() if int(metrics["support"]) > 0]
    return {
        "quartiles": bins,
        "support_vs_recall_delta_spearman": spearman(
            [float(metrics["support"]) for metrics in active],
            [float(metrics["delta_recall"]) for metrics in active],
        ),
        "support_vs_f1_delta_spearman": spearman(
            [float(metrics["support"]) for metrics in active],
            [float(metrics["delta_f1"]) for metrics in active],
        ),
    }


def _raw_leave_one_out_influence(
    scores: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(masked_scores, dim=-1) * key_mask.to(dtype=scores.dtype)
    output = torch.einsum("bhqk,bhkd->bhqd", attention, value)
    delta = (
        attention.unsqueeze(-1)
        / (1.0 - attention).clamp_min(0.05).unsqueeze(-1)
        * (value[:, :, None, :, :] - output[:, :, :, None, :])
    )
    return delta.norm(dim=-1)


def _evidence_components(
    kernel: torch.nn.Module,
    scores: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        query_features, token_features, strength, base_attention = kernel._causal_features(  # type: ignore[attr-defined]
            scores,
            query,
            key,
            value,
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )
        readouts = []
        for head_index in range(kernel.config.num_heads):  # type: ignore[attr-defined]
            query_state, token_state = kernel._states(  # type: ignore[attr-defined]
                query_features,
                token_features,
                layer_index=layer_index,
                head_index=head_index,
            )
            readouts.append(kernel._readout(query_state, token_state))  # type: ignore[attr-defined]
        readout = torch.stack(readouts, dim=1)
        influence = _raw_leave_one_out_influence(
            scores, value, batch["attention_mask"]
        )
        combined = readout * strength
    return {
        "base_attention": base_attention,
        "readout": readout,
        "normalized_strength": strength,
        "raw_influence": influence,
        "combined_evidence": combined,
    }


def _masked_key_mean(values: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    weights = query_mask[:, None, :, None].to(dtype=values.dtype)
    denominator = (weights.sum(dim=(1, 2)) * values.shape[1]).clamp_min(1.0)
    return (values * weights).sum(dim=(1, 2)) / denominator


def _sample_layer_metrics(
    *,
    components: Mapping[str, torch.Tensor],
    steered_scores: torch.Tensor,
    base_scores: torch.Tensor,
    score_gradient: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> list[dict[str, float | None]]:
    base_attention = components["base_attention"]
    steered_attention = _masked_attention(steered_scores, batch["attention_mask"])
    query_mask = batch["attention_mask"]
    context_mask = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    token_values = {
        "readout": _masked_key_mean(components["readout"], query_mask),
        "raw_influence": _masked_key_mean(components["raw_influence"], query_mask),
        "combined_evidence": _masked_key_mean(components["combined_evidence"], query_mask),
        "absolute_margin_gradient": _masked_key_mean(score_gradient.abs(), query_mask),
        "positive_margin_gradient": _masked_key_mean(score_gradient.clamp_min(0.0), query_mask),
        "attention_delta": _masked_key_mean(steered_attention - base_attention, query_mask),
    }
    probability_delta = (steered_attention - base_attention).abs()
    query_weights = query_mask[:, None, :].to(dtype=probability_delta.dtype)
    tv = 0.5 * probability_delta.sum(dim=-1)
    tv = (tv * query_weights).sum(dim=(1, 2)) / (
        query_weights.sum(dim=(1, 2)) * probability_delta.shape[1]
    ).clamp_min(1.0)
    first_order = (score_gradient * (steered_scores - base_scores)).sum(dim=(1, 2, 3))
    result: list[dict[str, float | None]] = []
    for row in range(base_scores.shape[0]):
        mask = context_mask[row]
        vectors = {
            name: values[row][mask].detach().float().cpu().tolist()
            for name, values in token_values.items()
        }
        attention_delta = vectors["attention_delta"]
        count = len(attention_delta)
        result.append(
            {
                "attention_total_variation": float(tv[row].item()),
                "first_order_gold_margin_delta": float(first_order[row].item()),
                "context_increased_fraction": (
                    sum(value > 1e-8 for value in attention_delta) / count if count else None
                ),
                "context_decreased_fraction": (
                    sum(value < -1e-8 for value in attention_delta) / count if count else None
                ),
                "mean_raw_influence": (
                    sum(vectors["raw_influence"]) / count if count else None
                ),
                "readout_vs_influence_spearman": spearman(
                    vectors["readout"], vectors["raw_influence"]
                ),
                "readout_vs_abs_margin_gradient_spearman": spearman(
                    vectors["readout"], vectors["absolute_margin_gradient"]
                ),
                "influence_vs_abs_margin_gradient_spearman": spearman(
                    vectors["raw_influence"], vectors["absolute_margin_gradient"]
                ),
                "combined_vs_abs_margin_gradient_spearman": spearman(
                    vectors["combined_evidence"], vectors["absolute_margin_gradient"]
                ),
                "combined_vs_positive_margin_gradient_spearman": spearman(
                    vectors["combined_evidence"], vectors["positive_margin_gradient"]
                ),
                "combined_vs_abs_margin_gradient_top20_overlap": topk_overlap(
                    vectors["combined_evidence"], vectors["absolute_margin_gradient"]
                ),
                "attention_delta_vs_margin_gradient_spearman": spearman(
                    vectors["attention_delta"], vectors["positive_margin_gradient"]
                ),
            }
        )
    return result


def _evaluate_baseline(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
    log_every_batches: int,
) -> dict[str, Any]:
    predictions: list[int] = []
    labels: list[int] = []
    margins: list[float] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for raw_batch in tracked_batches(
            loader,
            total_batches=len(loader),  # type: ignore[arg-type]
            stage="diagnostic_baseline",
            phase="validation_diagnostic",
            log_every_batches=log_every_batches,
        ):
            batch = move_batch(raw_batch, device)
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * len(batch["labels"])
            total_items += len(batch["labels"])
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            margins.extend(_gold_margin(logits, batch["labels"]).cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
        "margins": margins,
    }


def _evaluate_selector(
    *,
    selector: str,
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    artifacts: Any,
    baseline: Mapping[str, Any],
    collect_attribution: bool,
    log_every_batches: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    predictions: list[int] = []
    labels: list[int] = []
    margins: list[float] = []
    total_loss = 0.0
    total_items = 0
    item_offset = 0
    mechanism = MechanismAccumulator.create(model.config.num_layers)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    for raw_batch in tracked_batches(
        loader,
        total_batches=len(loader),  # type: ignore[arg-type]
        stage=f"diagnostic_{selector}",
        phase="validation_diagnostic",
        log_every_batches=log_every_batches,
    ):
        batch = move_batch(raw_batch, device)
        captures: dict[int, tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        adapter.attach(hook_config(batch))
        try:
            if collect_attribution:
                for layer_index, path in enumerate(model.score_module_paths):
                    module = resolve_module(model, path)

                    def capture(
                        _module: torch.nn.Module,
                        inputs: tuple[object, ...],
                        output: object,
                        index: int = layer_index,
                    ) -> None:
                        if (
                            len(inputs) != 4
                            or not all(isinstance(value, torch.Tensor) for value in inputs)
                            or not isinstance(output, torch.Tensor)
                        ):
                            raise TypeError("Q-VRES diagnostic requires score/query/key/value tensors")
                        captures[index] = (
                            tuple(value.detach() for value in inputs),  # type: ignore[arg-type]
                            output,
                        )

                    handles.append(module.register_forward_hook(capture))
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
        finally:
            for handle in handles:
                handle.remove()
            adapter.remove()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite logits for selector {selector}")
        loss = F.cross_entropy(logits, batch["labels"])
        batch_margins = _gold_margin(logits, batch["labels"])
        total_loss += float(loss.item()) * len(batch["labels"])
        total_items += len(batch["labels"])
        batch_predictions = logits.argmax(dim=-1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
        margins.extend(batch_margins.detach().cpu().tolist())
        if collect_attribution:
            if set(captures) != set(range(model.config.num_layers)):
                raise RuntimeError("not all score layers were captured")
            outputs = [captures[index][1] for index in range(model.config.num_layers)]
            gradients = torch.autograd.grad(batch_margins.sum(), outputs)
            relation_names = [
                artifacts.id_to_label[int(label)] for label in batch["labels"].cpu().tolist()
            ]
            for layer_index, gradient in enumerate(gradients):
                inputs, steered_scores = captures[layer_index]
                scores, query, key, value = inputs
                components = _evidence_components(
                    kernel, scores, query, key, value, batch, layer_index
                )
                rows = _sample_layer_metrics(
                    components=components,
                    steered_scores=steered_scores.detach(),
                    base_scores=scores,
                    score_gradient=gradient.detach(),
                    batch=batch,
                )
                for row_index, row in enumerate(rows):
                    global_index = item_offset + row_index
                    row["actual_gold_margin_delta"] = (
                        float(batch_margins[row_index].detach().item())
                        - float(baseline["margins"][global_index])
                    )
                    mechanism.add(layer_index, relation_names[row_index], row)
        item_offset += len(batch["labels"])
    metrics = classification_metrics(predictions, labels, len(artifacts.label_to_id))
    metrics["loss"] = total_loss / max(total_items, 1)
    evaluation = {
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
        "margins": margins,
    }
    if not collect_attribution:
        return evaluation, None
    baseline_predictions = baseline["predictions"]
    baseline_labels = baseline["labels"]
    mechanism_result = mechanism.result()
    mechanism_result["prediction_flips"] = {
        "correct_to_incorrect": sum(
            base == gold and current != gold
            for base, current, gold in zip(baseline_predictions, predictions, baseline_labels)
        ),
        "incorrect_to_correct": sum(
            base != gold and current == gold
            for base, current, gold in zip(baseline_predictions, predictions, baseline_labels)
        ),
        "prediction_changed": sum(
            base != current for base, current in zip(baseline_predictions, predictions)
        ),
    }
    return evaluation, mechanism_result


def _mean_layer_metric(mechanism: Mapping[str, Any], name: str) -> float | None:
    values = [
        layer["metrics"].get(name, {}).get("mean")
        for layer in mechanism.get("layers", [])
    ]
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def build_hypothesis_assessment(summary: Mapping[str, Any]) -> dict[str, Any]:
    selectors = summary["selectors"]
    q_row = selectors.get("q_causal_transport")
    key_row = selectors.get("q_causal_key_only")
    if q_row is None:
        return {"status": "not_evaluated", "reason": "q_causal_transport was not selected"}
    q_support = q_row["support_analysis"]["quartiles"]
    rare = q_support[0]["mean_recall_delta"] if q_support else None
    frequent_values = [
        row["mean_recall_delta"] for row in q_support[1:] if row["mean_recall_delta"] is not None
    ]
    frequent = sum(frequent_values) / len(frequent_values) if frequent_values else None
    checks: dict[str, bool | None] = {
        "q_causal_macro_recall_decreased": q_row["delta_vs_baseline"]["macro_recall"] < 0.0,
        "rarest_quartile_declined_more_than_others": (
            rare < frequent if rare is not None and frequent is not None else None
        ),
    }
    relation_associations: dict[str, float | None] = {}
    mechanism_relations = q_row.get("mechanism", {}).get("per_relation", {})
    aligned_relations = [
        relation
        for relation, metrics in q_row["per_relation"].items()
        if int(metrics["support"]) > 0 and relation in mechanism_relations
    ]
    influence_recall = spearman(
        [
            float(mechanism_relations[relation]["mean_raw_influence"]["mean"])
            for relation in aligned_relations
            if mechanism_relations[relation]["mean_raw_influence"]["mean"] is not None
        ],
        [
            float(q_row["per_relation"][relation]["delta_recall"])
            for relation in aligned_relations
            if mechanism_relations[relation]["mean_raw_influence"]["mean"] is not None
        ],
    )
    relation_associations["mean_raw_influence_vs_recall_delta_spearman"] = influence_recall
    checks["higher_value_influence_associated_with_recall_decline"] = (
        influence_recall <= -0.1 if influence_recall is not None else None
    )
    q_magnitude_alignment = _mean_layer_metric(
        q_row.get("mechanism", {}), "combined_vs_abs_margin_gradient_spearman"
    )
    q_directional_alignment = _mean_layer_metric(
        q_row.get("mechanism", {}), "combined_vs_positive_margin_gradient_spearman"
    )
    relation_associations["q_causal_combined_vs_abs_margin_gradient_mean"] = q_magnitude_alignment
    relation_associations[
        "q_causal_combined_vs_positive_margin_gradient_mean"
    ] = q_directional_alignment
    checks["q_causal_magnitude_alignment_exceeds_directional_alignment"] = (
        q_magnitude_alignment - q_directional_alignment >= 0.2
        if q_magnitude_alignment is not None and q_directional_alignment is not None
        else None
    )
    if key_row is not None:
        checks["key_only_macro_f1_exceeds_q_causal"] = (
            key_row["metrics"]["macro_f1"] > q_row["metrics"]["macro_f1"]
        )
        q_alignment = _mean_layer_metric(
            q_row.get("mechanism", {}), "combined_vs_abs_margin_gradient_spearman"
        )
        key_alignment = _mean_layer_metric(
            key_row.get("mechanism", {}), "combined_vs_abs_margin_gradient_spearman"
        )
        checks["key_only_evidence_alignment_exceeds_q_causal"] = (
            key_alignment > q_alignment
            if q_alignment is not None and key_alignment is not None
            else None
        )
        q_changes = q_row.get("mechanism", {}).get("prediction_flips", {}).get(
            "prediction_changed"
        )
        key_changes = key_row.get("mechanism", {}).get("prediction_flips", {}).get(
            "prediction_changed"
        )
        checks["q_causal_changes_more_predictions_than_key_only"] = (
            int(q_changes) > int(key_changes)
            if q_changes is not None and key_changes is not None
            else None
        )
    observed = [value for value in checks.values() if value is not None]
    positive = sum(bool(value) for value in observed)
    unsigned_risk_checks = (
        checks.get("q_causal_macro_recall_decreased"),
        checks.get("key_only_macro_f1_exceeds_q_causal"),
        checks.get("q_causal_magnitude_alignment_exceeds_directional_alignment"),
        checks.get("q_causal_changes_more_predictions_than_key_only"),
    )
    if all(value is True for value in unsigned_risk_checks):
        status = "consistent_with_unsigned_value_influence_risk"
        next_step = "repair_directionless_value_influence_and_retain_key_only_as_follow_up"
    elif positive >= 2:
        status = "mixed_correlational_support"
        next_step = "inspect_relation_rows_before_selecting_q_causal_or_key_only"
    else:
        status = "not_supported_by_aggregates"
        next_step = "do_not_attribute_failure_to_value_influence_without_new_evidence"
    return {
        "status": status,
        "checks": checks,
        "relation_level_associations": relation_associations,
        "minority_recall_subhypothesis": (
            "correlational_support"
            if checks["rarest_quartile_declined_more_than_others"] is True
            or checks["higher_value_influence_associated_with_recall_decline"] is True
            else "not_supported_by_current_split"
        ),
        "next_step": next_step,
        "interpretation_limit": (
            "Gradient and first-order margin quantities are local attribution proxies. "
            "They establish correlation, not a causal proof."
        ),
    }


def _public_selector_row(
    evaluation: Mapping[str, Any],
    baseline: Mapping[str, Any],
    id_to_label: Mapping[int, str],
    mechanism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    per_relation = relation_metrics(
        evaluation["predictions"], evaluation["labels"], id_to_label
    )
    baseline_relation = relation_metrics(
        baseline["predictions"], baseline["labels"], id_to_label
    )
    per_relation = add_relation_deltas(per_relation, baseline_relation)
    row = {
        "metrics": evaluation["metrics"],
        "delta_vs_baseline": {
            name: float(evaluation["metrics"][name]) - float(baseline["metrics"][name])
            for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
        },
        "per_relation": per_relation,
        "support_analysis": support_analysis(per_relation),
    }
    if mechanism is not None:
        row["mechanism"] = mechanism
    return row


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    selectors = summary["selectors"]
    lines = [
        "# Q-VRES Validation Diagnostic",
        "",
        f"Split: `{summary['split']}`",
        f"Records: `{summary['records']}`",
        f"Hypothesis assessment: `{summary['hypothesis_assessment']['status']}`",
        "",
        "| selector | macro recall | delta recall | macro F1 | delta F1 | correct->wrong | wrong->correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, row in selectors.items():
        flips = row.get("mechanism", {}).get("prediction_flips", {})
        lines.append(
            f"| {selector} | {row['metrics']['macro_recall']:.6f} | "
            f"{row['delta_vs_baseline']['macro_recall']:.6f} | "
            f"{row['metrics']['macro_f1']:.6f} | "
            f"{row['delta_vs_baseline']['macro_f1']:.6f} | "
            f"{flips.get('correct_to_incorrect', '-')} | "
            f"{flips.get('incorrect_to_correct', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Evidence And Intervention",
            "",
            "| selector | layer | evidence vs abs grad | evidence vs positive grad | attention delta vs positive grad | attention TV |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for selector, row in selectors.items():
        for layer in row.get("mechanism", {}).get("layers", []):
            metrics = layer["metrics"]

            def mean(name: str) -> str:
                value = metrics.get(name, {}).get("mean")
                return f"{value:.6f}" if value is not None else "-"

            lines.append(
                f"| {selector} | {layer['layer_index']} | "
                f"{mean('combined_vs_abs_margin_gradient_spearman')} | "
                f"{mean('combined_vs_positive_margin_gradient_spearman')} | "
                f"{mean('attention_delta_vs_margin_gradient_spearman')} | "
                f"{mean('attention_total_variation')} |"
            )
    q_row = selectors.get("q_causal_transport")
    if q_row is not None:
        lines.extend(
            [
                "",
                "## Largest Q-Causal Recall Drops",
                "",
                "| relation | support | delta recall | delta F1 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        sorted_relations = sorted(
            (
                item
                for item in q_row["per_relation"].items()
                if int(item[1]["support"]) > 0
            ),
            key=lambda item: (item[1]["delta_recall"], item[1]["delta_f1"]),
        )
        for relation, metrics in sorted_relations[:20]:
            lines.append(
                f"| {relation} | {metrics['support']} | "
                f"{metrics['delta_recall']:.6f} | {metrics['delta_f1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Suggested next step: `{summary['hypothesis_assessment']['next_step']}`",
            f"- Minority-recall subhypothesis: `{summary['hypothesis_assessment']['minority_recall_subhypothesis']}`",
            "- All method-selection evidence in this report comes from validation unless the report is explicitly marked test-only.",
            "- Evidence correlations and first-order margin deltas are attribution proxies, not causal proof.",
            "- The report contains no raw text, per-example predictions, token sequences, checkpoints, or gradient tensors.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_provenance(layout: RunLayout, split: str) -> dict[str, Any]:
    summary_path = layout.run_dir / "run_summary.json"
    if not summary_path.is_file():
        return {"available": False}
    summary = _read_json(summary_path)
    provenance = summary.get("provenance", {})
    return {
        "available": True,
        "seed": summary.get("seed")
        or next((row.get("seed") for row in summary.get("results", [])), None),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "config_sha256": provenance.get("config_sha256"),
        "split_source_sha256": provenance.get(split, {}).get("source_sha256"),
    }


def _metric_reproduction_check(
    layout: RunLayout,
    selectors: Mapping[str, Mapping[str, Any]],
    split: str,
    *,
    sampled: bool,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    summary_path = layout.run_dir / "run_summary.json"
    if sampled or not summary_path.is_file():
        return {
            "status": "not_applicable",
            "reason": "sampled diagnostic or source summary unavailable",
        }
    source = _read_json(summary_path)
    stored = {
        row["selector"]: row[split]["metrics"]
        for row in source.get("results", [])
        if isinstance(row, dict) and split in row
    }
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for selector, row in selectors.items():
        if selector not in stored:
            continue
        deltas = {
            name: float(row["metrics"][name]) - float(stored[selector][name])
            for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
        }
        maximum = max(abs(value) for value in deltas.values())
        comparisons[selector] = {"metric_deltas": deltas, "max_absolute_delta": maximum}
        if maximum > tolerance:
            failures.append(selector)
    return {
        "status": "pass" if comparisons and not failures else "fail",
        "tolerance": tolerance,
        "comparisons": comparisons,
        "failed_selectors": failures,
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.allow_test_report:
        raise ValueError("test diagnostics require --allow-test-report and must not drive method selection")
    selectors = [item.strip() for item in args.selectors.split(",") if item.strip()]
    attribution_selectors = {
        item.strip() for item in args.attribution_selectors.split(",") if item.strip()
    }
    if not selectors or len(selectors) != len(set(selectors)):
        raise ValueError("selectors must be non-empty and unique")
    if any(selector == "disabled" or selector not in SELECTORS for selector in selectors):
        raise ValueError("selectors must contain trainable Q-VRES selectors only")
    if not attribution_selectors.issubset(selectors):
        raise ValueError("attribution selectors must be included in selectors")
    layout = resolve_run_layout(args.run_dir, args.split)
    device = choose_device(args.device)
    artifacts = load_relation_run(layout.baseline_dir, device)
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    records: list[RelationRecord] = load_relation_jsonl(layout.data_path)
    source_record_count = len(records)
    if args.max_records > 0:
        records = sample_relation_records_proportional(
            records, args.max_records, seed=args.sample_seed
        )
    loader = make_relation_loader(
        records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    baseline = _evaluate_baseline(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        args.log_every_batches,
    )
    baseline_relation = relation_metrics(
        baseline["predictions"], baseline["labels"], artifacts.id_to_label
    )
    selector_rows: dict[str, Any] = {
        "disabled": {
            "metrics": baseline["metrics"],
            "delta_vs_baseline": {
                name: 0.0
                for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
            },
            "per_relation": add_relation_deltas(baseline_relation, baseline_relation),
            "support_analysis": support_analysis(
                add_relation_deltas(baseline_relation, baseline_relation)
            ),
        }
    }
    for selector in selectors:
        kernel = load_selector_kernel(selector, layout, artifacts.model, device)
        evaluation, mechanism = _evaluate_selector(
            selector=selector,
            model=artifacts.model,
            kernel=kernel,
            loader=loader,
            device=device,
            artifacts=artifacts,
            baseline=baseline,
            collect_attribution=selector in attribution_selectors,
            log_every_batches=args.log_every_batches,
        )
        selector_rows[selector] = _public_selector_row(
            evaluation, baseline, artifacts.id_to_label, mechanism
        )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        layout.run_dir
        / "diagnostics"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = len(records) < source_record_count
    summary: dict[str, Any] = {
        "schema_version": "q-attention.q-vres.validation-diagnostic.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_name": layout.run_dir.name,
        "parallel_mode": layout.parallel_mode,
        "split": args.split,
        "selection_policy": (
            "validation_only_method_diagnosis"
            if args.split == "valid"
            else "test_reporting_only_not_for_method_selection"
        ),
        "records": len(records),
        "sampled": sampled,
        "device": str(device),
        "source_provenance": _source_provenance(layout, args.split),
        "diagnostic_code": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_dirty": bool(git_output("status", "--porcelain")),
        },
        "selectors": selector_rows,
        "privacy": {
            "contains_raw_text": False,
            "contains_per_example_predictions": False,
            "contains_tokens": False,
            "contains_checkpoints": False,
            "contains_gradient_tensors": False,
        },
        "methodological_limits": {
            "gradient_attribution_is_causal_proof": False,
            "first_order_margin_delta_is_causal_proof": False,
            "test_metrics_can_drive_method_selection": False,
        },
        "attribution_definitions": {
            "raw_value_influence": (
                "norm of the exact leave-one-out attention-output delta before normalization"
            ),
            "quantum_readout": "squared quantum-state overlap used by the selector",
            "combined_evidence": "quantum readout times normalized value influence",
            "absolute_margin_gradient": (
                "absolute derivative of post-intervention gold margin with respect to steered scores"
            ),
            "positive_margin_gradient": (
                "positive part of the post-intervention gold-margin score derivative"
            ),
            "first_order_gold_margin_delta": (
                "sum of score gradient times the applied score residual"
            ),
            "attention_total_variation": (
                "mean half-L1 distance between pre- and post-intervention attention rows"
            ),
        },
    }
    summary["metric_reproduction"] = _metric_reproduction_check(
        layout, selector_rows, args.split, sampled=sampled
    )
    if summary["metric_reproduction"]["status"] == "fail":
        raise RuntimeError(
            "diagnostic predictions do not reproduce the source run metrics; "
            "check code, checkpoints, data, and selector configuration"
        )
    summary["hypothesis_assessment"] = build_hypothesis_assessment(summary)
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(summary, output_dir / "diagnostic_summary.md")
    print(
        json.dumps(
            {
                "event": "diagnostic_complete",
                "output_dir": str(output_dir),
                "records": len(records),
                "hypothesis_status": summary["hypothesis_assessment"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose relation-level and attention-level Q-VRES behavior from an existing raw run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--allow-test-report", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument(
        "--selectors",
        default=",".join(SELECTORS[1:]),
    )
    parser.add_argument(
        "--attribution-selectors",
        default=",".join(DEFAULT_ATTRIBUTION_SELECTORS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.log_every_batches <= 0:
        raise SystemExit("batch size and log interval must be positive")
    if args.max_records < 0:
        raise SystemExit("max records must be non-negative; zero means the full split")
    run_diagnostics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
