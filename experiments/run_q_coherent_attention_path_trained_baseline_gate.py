#!/usr/bin/env python3
"""Gate Q-WAP on attention scores learned by a complete relation model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.models import RelationExtractionModel, RelationTransformerConfig  # noqa: E402
from q_attention.plugins.q_coherent_attention_path import (  # noqa: E402
    CoherentAttentionPathConfig,
    CoherentAttentionPathKernel,
    build_coherent_attention_path_kernel,
)


ROLE_IDS = {
    "anchor": 1,
    "subject": 2,
    "object": 3,
    "candidate_positive": 4,
    "candidate_negative": 5,
    "bridge": 6,
}
NUISANCE_START = 7
SELECTORS = (
    "disabled",
    "q_wap_signed",
    "q_wap_unsigned",
    "classical_wap_diffusion",
    "direct_row",
    "shuffled_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_coherent_attention_path_trained_baseline_gate.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.q-wap-trained-baseline-gate.v1":
        raise ValueError("unsupported trained-baseline Q-WAP config")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("trained-baseline Q-WAP gate requires seed 7")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("selectors must match the frozen trained-baseline allowlist")
    if abs(float(config["mechanism"]["walk_time"]) - math.pi / 4.0) > 1e-12:
        raise ValueError("walk time must remain pi/4")
    streams = [int(config["dataset"][f"{name}_stream"]) for name in ("train", "valid", "test")]
    if len(set(streams)) != 3:
        raise ValueError("train, valid, and test streams must be distinct")
    if int(config["dataset"]["nodes"]) != 7:
        raise ValueError("the structural relation task requires seven nodes")
    if config["training"].get("optimizer") != "adam_full_batch":
        raise ValueError("selector training must use the frozen full-batch optimizer")
    return config


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_split(
    stream: int,
    size: int,
    nuisance_tokens: int,
    device: torch.device,
    seen: set[tuple[int, ...]] | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    if size <= 0 or size % 2:
        raise ValueError("split size must be positive and even")
    if nuisance_tokens < 8:
        raise ValueError("at least eight nuisance tokens are required")
    generator = torch.Generator(device="cpu").manual_seed(stream)
    target_labels = torch.arange(size, dtype=torch.long) % 2
    target_labels = target_labels[torch.randperm(size, generator=generator)]
    seen = seen if seen is not None else set()
    rows: list[torch.Tensor] = []
    subject_masks: list[torch.Tensor] = []
    object_masks: list[torch.Tensor] = []
    fingerprints: list[str] = []
    base = [
        ROLE_IDS["anchor"],
        ROLE_IDS["subject"],
        ROLE_IDS["object"],
        ROLE_IDS["candidate_positive"],
        ROLE_IDS["candidate_negative"],
        ROLE_IDS["bridge"],
    ]
    for target in target_labels.tolist():
        for _attempt in range(10000):
            nuisance = NUISANCE_START + int(
                torch.randint(nuisance_tokens, (1,), generator=generator)
            )
            permutation = torch.randperm(7, generator=generator)
            row = torch.tensor(base + [nuisance], dtype=torch.long)[permutation]
            subject_index = int((row == ROLE_IDS["subject"]).nonzero()[0])
            object_index = int((row == ROLE_IDS["object"]).nonzero()[0])
            positive_index = int((row == ROLE_IDS["candidate_positive"]).nonzero()[0])
            negative_index = int((row == ROLE_IDS["candidate_negative"]).nonzero()[0])
            label = int((subject_index < object_index) ^ (positive_index < negative_index))
            key = tuple(int(value) for value in row)
            if label != target or key in seen:
                continue
            seen.add(key)
            subject_mask = torch.zeros(7, dtype=torch.bool)
            object_mask = torch.zeros(7, dtype=torch.bool)
            subject_mask[subject_index] = True
            object_mask[object_index] = True
            rows.append(row)
            subject_masks.append(subject_mask)
            object_masks.append(object_mask)
            fingerprints.append(hashlib.sha256(bytes(key)).hexdigest())
            break
        else:
            raise RuntimeError("could not construct a unique balanced structural example")
    input_ids = torch.stack(rows).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "subject_mask": torch.stack(subject_masks).to(device),
        "object_mask": torch.stack(object_masks).to(device),
        "labels": target_labels.to(device),
        "fingerprints": fingerprints,
    }


def make_splits(config: dict[str, Any], device: torch.device) -> dict[str, dict[str, Any]]:
    dataset = config["dataset"]
    seen: set[tuple[int, ...]] = set()
    result: dict[str, dict[str, Any]] = {}
    for name in ("train", "valid", "test"):
        result[name] = make_split(
            int(dataset[f"{name}_stream"]),
            int(dataset[f"{name}_size"]),
            int(dataset["nuisance_tokens"]),
            device,
            seen,
        )
    return result


def tensor_batch(split: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in split.items() if isinstance(value, torch.Tensor)}


def shuffled_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if batch["input_ids"].shape[0] < 2:
        raise ValueError("shuffled-anchor control requires at least two examples")
    return {
        **batch,
        "subject_mask": batch["subject_mask"].roll(1, dims=0),
        "object_mask": batch["object_mask"].roll(1, dims=0),
    }


def model_forward(model: RelationExtractionModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["subject_mask"],
        batch["object_mask"],
    )


def metric_row(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prediction = logits.argmax(dim=-1)
    return {
        "accuracy": float(prediction.eq(labels).float().mean()),
        "nll": float(F.cross_entropy(logits, labels)),
        "mean_margin": float(
            (logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(
                F.one_hot(labels, logits.shape[-1]).bool(), torch.finfo(logits.dtype).min
            ).max(dim=-1).values).mean()
        ),
        "predictions": prediction.detach(),
    }


def evaluate_baseline(
    model: RelationExtractionModel, split: dict[str, Any]
) -> tuple[dict[str, Any], torch.Tensor]:
    batch = tensor_batch(split)
    model.eval()
    with torch.no_grad():
        logits = model_forward(model, batch)
    return metric_row(logits, batch["labels"]), logits


def train_baseline(
    splits: dict[str, dict[str, Any]], config: dict[str, Any], device: torch.device
) -> tuple[RelationExtractionModel, dict[str, Any]]:
    model_config = config["model"]
    dataset = config["dataset"]
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=max(
                int(tensor_batch(split)["input_ids"].max())
                for split in splits.values()
            )
            + 1,
            num_labels=2,
            dim=int(model_config["dim"]),
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ff_dim=int(model_config["ff_dim"]),
            dropout=float(model_config["dropout"]),
            max_length=int(
                tensor_batch(next(iter(splits.values())))["input_ids"].shape[1]
            ),
        )
    ).to(device)
    train = tensor_batch(splits["train"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"])
    )
    batch_size = int(model_config["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_key = (float("-inf"), float("-inf"))
    for epoch in range(1, int(model_config["epochs"]) + 1):
        model.train()
        order = torch.randperm(train["labels"].shape[0], generator=generator)
        losses: list[float] = []
        for start in range(0, order.shape[0], batch_size):
            index = order[start : start + batch_size].to(device)
            batch = {key: value[index] for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model_forward(model, batch)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        valid_metrics, _ = evaluate_baseline(model, splits["valid"])
        record = {
            "epoch": epoch,
            "train_nll": sum(losses) / len(losses),
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_nll": valid_metrics["nll"],
        }
        history.append(record)
        key = (valid_metrics["accuracy"], -valid_metrics["nll"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("baseline training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metrics = {}
    logits = {}
    for name in ("train", "valid", "test"):
        metrics[name], logits[name] = evaluate_baseline(model, splits[name])
        metrics[name].pop("predictions")
    return model, {
        "best_epoch": best_epoch,
        "history": history,
        "metrics": metrics,
        "logits": logits,
    }


@contextmanager
def capture_score_tensors(
    model: RelationExtractionModel,
) -> Iterator[list[dict[str, torch.Tensor]]]:
    captured: list[dict[str, torch.Tensor]] = [{} for _ in model.score_module_paths]
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            if len(inputs) != 4 or not all(isinstance(item, torch.Tensor) for item in inputs):
                raise TypeError("score hook must receive score, query, key, and value tensors")
            if not isinstance(output, torch.Tensor):
                raise TypeError("score hook output must be a tensor")
            scores, query, key, value = inputs
            captured[layer_index] = {
                "scores": scores.detach(),
                "query": query.detach(),
                "key": key.detach(),
                "value": value.detach(),
            }
            return output

        return hook

    try:
        for layer_index, layer in enumerate(model.encoder.layers):
            handles.append(layer.attn.score_intervention.register_forward_hook(make_hook(layer_index)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def collect_scores(
    model: RelationExtractionModel, split: dict[str, Any]
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    batch = tensor_batch(split)
    with torch.no_grad(), capture_score_tensors(model) as captured:
        logits = model_forward(model, batch)
    if any(not row for row in captured):
        raise RuntimeError("one or more score hooks did not capture tensors")
    return captured, logits


def _graph(scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    pair = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
    graph = 0.5 * (scores + scores.transpose(-1, -2))
    graph = graph * pair.to(graph.dtype)
    return graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))


def _quantum_probabilities(graph: torch.Tensor, walk_time: float) -> torch.Tensor:
    complex_dtype = torch.complex128 if graph.dtype == torch.float64 else torch.complex64
    return torch.matrix_exp((-1j * walk_time) * graph.to(complex_dtype)).abs().square().to(graph.dtype)


def geometry_diagnostics(
    captures: list[dict[str, torch.Tensor]], batch: dict[str, torch.Tensor], walk_time: float
) -> dict[str, Any]:
    cycle_products: list[torch.Tensor] = []
    probability_effects: list[torch.Tensor] = []
    asymmetries: list[torch.Tensor] = []
    for row in captures:
        scores = row["scores"]
        graph = _graph(scores, batch["attention_mask"])
        asymmetries.append((scores - scores.transpose(-1, -2)).abs().flatten())
        signed = _quantum_probabilities(graph, walk_time)
        unsigned = _quantum_probabilities(graph.abs(), walk_time)
        probability_effects.append(0.5 * (signed - unsigned).abs().sum(dim=-1).flatten())
        for i, j, k in itertools.combinations(range(scores.shape[-1]), 3):
            cycle_products.append(graph[..., i, j] * graph[..., j, k] * graph[..., k, i])
    products = torch.cat([item.flatten() for item in cycle_products])
    nonzero = products.abs() > 1e-6
    effects = torch.cat(probability_effects)
    asymmetry = torch.cat(asymmetries)
    return {
        "nonzero_cycle_fraction": float(nonzero.float().mean()),
        "negative_signed_cycle_fraction": float((products[nonzero] < 0).float().mean()) if nonzero.any() else 0.0,
        "signed_unsigned_probability_tv_mean": float(effects.mean()),
        "signed_unsigned_probability_tv_max": float(effects.max()),
        "raw_score_asymmetry_mean": float(asymmetry.mean()),
        "raw_score_asymmetry_max": float(asymmetry.max()),
    }


@contextmanager
def fixed_score_residuals(
    model: RelationExtractionModel, residuals: list[torch.Tensor]
) -> Iterator[None]:
    handles = []
    for layer_index, layer in enumerate(model.encoder.layers):
        handles.append(
            layer.attn.score_intervention.register_forward_hook(
                lambda _module, _inputs, output, idx=layer_index: output + residuals[idx]
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def oracle_action_headroom(
    model: RelationExtractionModel, split: dict[str, Any], step: float
) -> dict[str, Any]:
    batch = tensor_batch(split)
    leaves: list[torch.Tensor | None] = [None] * len(model.encoder.layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> torch.Tensor:
            if not isinstance(output, torch.Tensor):
                raise TypeError("score hook output must be a tensor")
            leaf = output.detach().requires_grad_(True)
            leaves[layer_index] = leaf
            return leaf

        return hook

    try:
        for layer_index, layer in enumerate(model.encoder.layers):
            handles.append(layer.attn.score_intervention.register_forward_hook(make_hook(layer_index)))
        logits = model_forward(model, batch)
        baseline_loss = F.cross_entropy(logits, batch["labels"])
        baseline_loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    context = batch["attention_mask"] & ~(batch["subject_mask"] | batch["object_mask"])
    context_mask = context[:, None, None, :]
    residuals = []
    gradient_norms = []
    for leaf in leaves:
        if leaf is None or leaf.grad is None:
            raise RuntimeError("score-space oracle did not receive gradients")
        gradient = leaf.grad.detach()
        gradient_norms.append(float(torch.linalg.vector_norm(gradient)))
        masked = gradient * context_mask.to(gradient.dtype)
        count = context_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        centered = masked - (masked.sum(dim=-1, keepdim=True) / count) * context_mask
        scale = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        residuals.append((-step * centered / scale).detach())
    with torch.no_grad(), fixed_score_residuals(model, residuals):
        oracle_logits = model_forward(model, batch)
    oracle_loss = F.cross_entropy(oracle_logits, batch["labels"])
    return {
        "baseline_nll": float(baseline_loss.detach()),
        "oracle_nll": float(oracle_loss),
        "nll_headroom": float(baseline_loss.detach() - oracle_loss),
        "score_gradient_norms": gradient_norms,
        "maximum_residual": max(float(row.abs().max()) for row in residuals),
    }


class DirectRowKernel(CoherentAttentionPathKernel):
    """Matched row-local control without path propagation."""

    kernel_type = "direct_row"

    def _path_probabilities(self, graph: torch.Tensor) -> torch.Tensor:
        return torch.softmax(graph, dim=-1)


class SignedRelationContrastKernel(CoherentAttentionPathKernel):
    """Use subject-object path contrast as a directed relation residual."""

    def __init__(
        self, config: CoherentAttentionPathConfig, path_type: str
    ) -> None:
        self.kernel_type = path_type
        super().__init__(config)
        ratio = config.initial_transport / config.max_transport
        initial_raw = math.atanh(ratio)
        self.raw_transport = nn.Parameter(
            torch.full((config.num_layers, config.num_heads), initial_raw)
        )

    def transport_fractions(self, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= self.config.num_layers:
            raise ValueError("layer_index is outside the configured model")
        return self.config.max_transport * torch.tanh(
            self.raw_transport[layer_index]
        )

    def _path_probabilities(self, graph: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "direct_row":
            return torch.softmax(graph, dim=-1)
        return super()._path_probabilities(graph)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        scores: torch.Tensor,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del query, key, value
        self._validate(scores, attention_mask, subject_mask, object_mask)
        graph = self._hermitian_graph(scores, attention_mask)
        path = self._path_probabilities(graph)
        subject_query = subject_mask[:, None, :, None].to(scores.dtype)
        object_query = object_mask[:, None, :, None].to(scores.dtype)
        subject_path = (path * subject_query).sum(dim=-2) / subject_query.sum(
            dim=-2
        ).clamp_min(1.0)
        object_path = (path * object_query).sum(dim=-2) / object_query.sum(
            dim=-2
        ).clamp_min(1.0)
        context = attention_mask & ~(subject_mask | object_mask)
        context_keys = context[:, None, :].to(scores.dtype)
        contrast = (subject_path - object_path) * context_keys
        context_count = context_keys.sum(dim=-1, keepdim=True).clamp_min(1.0)
        contrast = contrast - (
            contrast.sum(dim=-1, keepdim=True) / context_count
        ) * context_keys
        contrast = contrast / contrast.abs().sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        query_sign = subject_query - object_query
        fraction = self.transport_fractions(layer_index).view(1, -1, 1, 1)
        residual = fraction * query_sign * contrast[:, :, None, :]
        return residual.to(dtype=scores.dtype)


class DirectedRelationContrastKernel(SignedRelationContrastKernel):
    """Quantum relation contrast retaining Q/K direction as complex phase."""

    kernel_type = "quantum_directed"

    def __init__(self, config: CoherentAttentionPathConfig) -> None:
        super().__init__(config, "quantum_directed")

    def _hermitian_graph(
        self, scores: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        node_mask = attention_mask.to(dtype=torch.bool)
        pair_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
        symmetric = 0.5 * (scores + scores.transpose(-1, -2))
        skew = 0.5 * (scores - scores.transpose(-1, -2))
        complex_dtype = (
            torch.complex128 if scores.dtype == torch.float64 else torch.complex64
        )
        graph = symmetric.to(complex_dtype) + 1j * skew.to(complex_dtype)
        graph = graph * pair_mask.to(graph.dtype)
        return graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))

    def _path_probabilities(self, graph: torch.Tensor) -> torch.Tensor:
        unitary = torch.matrix_exp((-1j * self.config.walk_time) * graph)
        return unitary.abs().square().to(graph.real.dtype)


class ChiralRelationContrastKernel(DirectedRelationContrastKernel):
    """Pure directed quantum walk from the antisymmetric Q/K score flow."""

    kernel_type = "quantum_chiral"

    def _hermitian_graph(
        self, scores: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        node_mask = attention_mask.to(dtype=torch.bool)
        pair_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
        skew = 0.5 * (scores - scores.transpose(-1, -2))
        complex_dtype = (
            torch.complex128 if scores.dtype == torch.float64 else torch.complex64
        )
        graph = 1j * skew.to(complex_dtype) * pair_mask.to(complex_dtype)
        return graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))


class RawDirectRelationContrastKernel(SignedRelationContrastKernel):
    """Row-local directional control using the original non-symmetric scores."""

    def __init__(self, config: CoherentAttentionPathConfig) -> None:
        super().__init__(config, "direct_row")

    def _hermitian_graph(
        self, scores: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pair_mask = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
        graph = scores * pair_mask.to(scores.dtype)
        return graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))


def build_selector(selector: str, model: RelationExtractionModel, config: dict[str, Any]):
    if selector == "disabled":
        return None
    mechanism = config["mechanism"]
    kernel_config = CoherentAttentionPathConfig(
        num_layers=model.config.num_layers,
        num_heads=model.config.num_heads,
        max_transport=float(mechanism["max_transport"]),
        initial_transport=float(mechanism["initial_transport"]),
        walk_time=float(mechanism["walk_time"]),
    )
    readout = mechanism.get("readout", "row_transport")
    if readout == "chiral_relation_contrast":
        if selector in {"q_wap_signed", "shuffled_anchor"}:
            return ChiralRelationContrastKernel(kernel_config)
        if selector == "direct_row":
            return RawDirectRelationContrastKernel(kernel_config)
        path_types = {
            "q_wap_unsigned": "quantum_unsigned",
            "classical_wap_diffusion": "classical_diffusion",
        }
        return SignedRelationContrastKernel(kernel_config, path_types[selector])
    if readout == "directed_relation_contrast":
        if selector in {"q_wap_signed", "shuffled_anchor"}:
            return DirectedRelationContrastKernel(kernel_config)
        if selector == "direct_row":
            return RawDirectRelationContrastKernel(kernel_config)
        path_types = {
            "q_wap_unsigned": "quantum_unsigned",
            "classical_wap_diffusion": "classical_diffusion",
        }
        return SignedRelationContrastKernel(kernel_config, path_types[selector])
    if readout == "relation_contrast":
        path_types = {
            "q_wap_signed": "quantum_signed",
            "shuffled_anchor": "quantum_signed",
            "q_wap_unsigned": "quantum_unsigned",
            "classical_wap_diffusion": "classical_diffusion",
            "direct_row": "direct_row",
        }
        return SignedRelationContrastKernel(kernel_config, path_types[selector])
    if selector in {"q_wap_signed", "shuffled_anchor"}:
        return build_coherent_attention_path_kernel("quantum_signed", kernel_config)
    if selector == "q_wap_unsigned":
        return build_coherent_attention_path_kernel("quantum_unsigned", kernel_config)
    if selector == "classical_wap_diffusion":
        return build_coherent_attention_path_kernel("classical_diffusion", kernel_config)
    if selector == "direct_row":
        return DirectRowKernel(kernel_config)
    raise ValueError(f"unknown selector: {selector}")


@contextmanager
def score_intervention(
    model: RelationExtractionModel,
    kernel: CoherentAttentionPathKernel | None,
    batch: dict[str, torch.Tensor],
    *,
    shuffle_anchors: bool = False,
    query_scope: str = "all",
) -> Iterator[None]:
    if kernel is None:
        yield
        return
    hook_batch = shuffled_batch(batch) if shuffle_anchors else batch
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> torch.Tensor:
            if len(inputs) != 4 or not all(isinstance(item, torch.Tensor) for item in inputs):
                raise TypeError("score hook must receive score, query, key, and value tensors")
            if not isinstance(output, torch.Tensor):
                raise TypeError("score hook output must be a tensor")
            scores, query, key, value = inputs
            residual = kernel(
                query,
                key,
                value,
                scores=scores,
                layer_index=layer_index,
                attention_mask=hook_batch["attention_mask"],
                subject_mask=hook_batch["subject_mask"],
                object_mask=hook_batch["object_mask"],
            )
            if query_scope == "subject_object":
                query_mask = (
                    hook_batch["subject_mask"] | hook_batch["object_mask"]
                )[:, None, :, None]
                residual = residual * query_mask.to(residual.dtype)
            elif query_scope != "all":
                raise ValueError(f"unsupported query scope: {query_scope}")
            return output + residual

        return hook

    try:
        for layer_index, layer in enumerate(model.encoder.layers):
            handles.append(layer.attn.score_intervention.register_forward_hook(make_hook(layer_index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def selector_metrics(
    model: RelationExtractionModel,
    kernel: CoherentAttentionPathKernel | None,
    split: dict[str, Any],
    baseline_prediction: torch.Tensor,
    selector: str,
    query_scope: str,
) -> dict[str, Any]:
    batch = tensor_batch(split)
    with torch.no_grad(), score_intervention(
        model,
        kernel,
        batch,
        shuffle_anchors=selector == "shuffled_anchor",
        query_scope=query_scope,
    ):
        logits = model_forward(model, batch)
    row = metric_row(logits, batch["labels"])
    prediction = row.pop("predictions")
    return {
        **row,
        "corrected_examples": int((~baseline_prediction.eq(batch["labels"]) & prediction.eq(batch["labels"])).sum()),
        "harmed_correct_examples": int((baseline_prediction.eq(batch["labels"]) & ~prediction.eq(batch["labels"])).sum()),
    }


def train_selector(
    selector: str,
    model: RelationExtractionModel,
    splits: dict[str, dict[str, Any]],
    baseline_predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    kernel = build_selector(selector, model, config)
    if kernel is not None:
        kernel = kernel.to(device)
    query_scope = str(config["mechanism"].get("query_scope", "all"))
    initial = {
        name: selector_metrics(
            model,
            kernel,
            splits[name],
            baseline_predictions[name],
            selector,
            query_scope,
        )
        for name in ("train", "valid", "test")
    }
    losses = [initial["train"]["nll"]]
    if kernel is not None:
        optimizer = torch.optim.Adam(
            kernel.parameters(), lr=float(config["training"]["learning_rate"])
        )
        train_batch = tensor_batch(splits["train"])
        for _ in range(int(config["training"]["steps"])):
            optimizer.zero_grad(set_to_none=True)
            with score_intervention(
                model,
                kernel,
                train_batch,
                shuffle_anchors=selector == "shuffled_anchor",
                query_scope=query_scope,
            ):
                logits = model_forward(model, train_batch)
            loss = F.cross_entropy(logits, train_batch["labels"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    final = {
        name: selector_metrics(
            model,
            kernel,
            splits[name],
            baseline_predictions[name],
            selector,
            query_scope,
        )
        for name in ("train", "valid", "test")
    }
    parameters = 0 if kernel is None else sum(parameter.numel() for parameter in kernel.parameters())
    transports = [] if kernel is None else kernel.transport_fractions(0).detach().cpu().tolist()
    return {
        "selector": selector,
        "trainable_parameters": parameters,
        "initial": initial,
        "final": final,
        "training": {
            "steps": 0 if kernel is None else int(config["training"]["steps"]),
            "initial_nll": losses[0],
            "final_step_nll": losses[-1],
            "minimum_nll": min(losses),
            "finite": all(math.isfinite(value) for value in losses),
            "layer0_transport": transports,
        },
    }


def stage_a_gate(
    baseline: dict[str, Any], diagnostics: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["stage_a_gate"]
    valid_geometry = diagnostics["geometry"]["valid"]
    checks = {
        "disjoint_splits": diagnostics["exact_split_overlap"] == 0,
        "balanced_splits": all(value == 0.5 for value in diagnostics["positive_label_fraction"].values()),
        "baseline_valid": baseline["metrics"]["valid"]["accuracy"] >= float(gate["minimum_valid_accuracy"]),
        "baseline_test": baseline["metrics"]["test"]["accuracy"] >= float(gate["minimum_test_accuracy"]),
        "disabled_parity": diagnostics["maximum_disabled_logit_difference"] <= float(gate["maximum_parity_error"]),
        "nontrivial_asymmetry": valid_geometry["raw_score_asymmetry_mean"] >= float(gate["minimum_score_asymmetry"]),
        "signed_cycles_present": valid_geometry["negative_signed_cycle_fraction"] >= float(gate["minimum_negative_cycle_fraction"]),
        "signed_effect_present": valid_geometry["signed_unsigned_probability_tv_mean"] >= float(gate["minimum_signed_unsigned_tv"]),
        "valid_action_headroom": diagnostics["action_headroom"]["valid"]["nll_headroom"] >= float(gate["minimum_oracle_nll_headroom"]),
        "test_action_headroom": diagnostics["action_headroom"]["test"]["nll_headroom"] >= float(gate["minimum_oracle_nll_headroom"]),
    }
    if not checks["baseline_valid"] or not checks["baseline_test"]:
        failure_reason = "baseline_invalid"
    elif not checks["signed_cycles_present"] or not checks["signed_effect_present"]:
        failure_reason = "signed_cycles_sparse_or_degenerate"
    elif not checks["valid_action_headroom"] or not checks["test_action_headroom"]:
        failure_reason = "no_score_action_headroom"
    elif not all(checks.values()):
        failure_reason = "stage_a_invariant_failure"
    else:
        failure_reason = None
    passed = all(checks.values())
    return {
        **checks,
        "status": "pass" if passed else "fail",
        "failure_reason": failure_reason,
        "stage_b_authorized": passed,
    }


def stage_b_gate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by = {row["selector"]: row for row in results}
    disabled = by["disabled"]
    quantum = by["q_wap_signed"]
    controls = [by[name] for name in ("q_wap_unsigned", "classical_wap_diffusion", "direct_row", "shuffled_anchor")]
    gate = config["stage_b_gate"]
    gains: dict[str, dict[str, float]] = {}
    for name, row in by.items():
        gains[name] = {
            split: disabled["final"][split]["nll"] - row["final"][split]["nll"]
            for split in ("valid", "test")
        }
    best_control = {
        split: min(row["final"][split]["nll"] for row in controls)
        for split in ("valid", "test")
    }
    advantage = {
        split: best_control[split] - quantum["final"][split]["nll"]
        for split in ("valid", "test")
    }
    matched = [row["trainable_parameters"] for row in results if row["selector"] != "disabled"]
    checks = {
        "parameter_matching": len(set(matched)) == 1 and matched[0] > 0,
        "finite_training": all(row["training"]["finite"] for row in results),
        "quantum_valid_gain": gains["q_wap_signed"]["valid"] >= float(gate["minimum_quantum_nll_gain"]),
        "quantum_test_gain": gains["q_wap_signed"]["test"] >= float(gate["minimum_quantum_nll_gain"]),
        "valid_control_advantage": advantage["valid"] >= float(gate["minimum_control_advantage"]),
        "test_control_advantage": advantage["test"] >= float(gate["minimum_control_advantage"]),
        "valid_accuracy_preserved": quantum["final"]["valid"]["accuracy"] + float(gate["maximum_accuracy_drop"]) >= disabled["final"]["valid"]["accuracy"],
        "test_accuracy_preserved": quantum["final"]["test"]["accuracy"] + float(gate["maximum_accuracy_drop"]) >= disabled["final"]["test"]["accuracy"],
    }
    if not checks["quantum_valid_gain"] or not checks["quantum_test_gain"]:
        failure_reason = "signed_effect_not_target_aligned"
    elif not checks["valid_control_advantage"] or not checks["test_control_advantage"]:
        unsigned_gain = max(gains["q_wap_unsigned"].values())
        shuffled_gain = max(gains["shuffled_anchor"].values())
        if shuffled_gain >= min(gains["q_wap_signed"].values()):
            failure_reason = "shuffled_anchor_insensitive"
        elif unsigned_gain >= min(gains["q_wap_signed"].values()):
            failure_reason = "unsigned_or_classical_parity"
        else:
            failure_reason = "matched_control_parity"
    elif not all(checks.values()):
        failure_reason = "stage_b_invariant_failure"
    else:
        failure_reason = None
    passed = all(checks.values())
    return {
        **checks,
        "status": "pass" if passed else "fail",
        "failure_reason": failure_reason,
        "nll_gains": gains,
        "quantum_over_best_control_nll_advantage": advantage,
        "multi_seed_authorized": passed,
        "real_data_authorized": False,
        "hardware_claim_authorized": False,
    }


def split_diagnostics(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fingerprint_sets = {name: set(split["fingerprints"]) for name, split in splits.items()}
    overlap = sum(
        len(fingerprint_sets[left] & fingerprint_sets[right])
        for left, right in itertools.combinations(fingerprint_sets, 2)
    )
    return {
        "exact_split_overlap": overlap,
        "positive_label_fraction": {
            name: float(tensor_batch(split)["labels"].float().mean())
            for name, split in splits.items()
        },
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    set_seed(int(config["seed"]))
    splits = make_splits(config, device)
    model, baseline = train_baseline(splits, config, device)
    diagnostics = split_diagnostics(splits)
    diagnostics["geometry"] = {}
    replay_logits = {}
    for name in ("valid", "test"):
        captures, replay_logits[name] = collect_scores(model, splits[name])
        diagnostics["geometry"][name] = geometry_diagnostics(
            captures,
            tensor_batch(splits[name]),
            float(config["mechanism"]["walk_time"]),
        )
    diagnostics["maximum_disabled_logit_difference"] = max(
        float((baseline["logits"][name] - replay_logits[name]).abs().max())
        for name in replay_logits
    )
    diagnostics["action_headroom"] = {
        name: oracle_action_headroom(
            model, splits[name], float(config["stage_a_gate"]["oracle_residual_step"])
        )
        for name in ("valid", "test")
    }
    stage_a = stage_a_gate(baseline, diagnostics, config)
    baseline_predictions = {
        name: baseline["logits"][name].argmax(dim=-1)
        for name in ("train", "valid", "test")
    }
    results: list[dict[str, Any]] = []
    stage_b: dict[str, Any] = {
        "status": "not_run",
        "failure_reason": "stage_a_failed",
        "multi_seed_authorized": False,
        "real_data_authorized": False,
        "hardware_claim_authorized": False,
    }
    if stage_a["stage_b_authorized"]:
        for selector in SELECTORS:
            results.append(
                train_selector(
                    selector,
                    model,
                    splits,
                    baseline_predictions,
                    config,
                    device,
                )
            )
        stage_b = stage_b_gate(results, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "baseline_model.pt"
    torch.save(model.state_dict(), checkpoint)
    baseline.pop("logits")
    summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "revision": git_revision(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "baseline_checkpoint_sha256": sha256(checkpoint),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "seed": int(config["seed"]),
        "dataset_identity": config["dataset"]["identity"],
        "baseline": baseline,
        "diagnostics": diagnostics,
        "stage_a_gate": stage_a,
        "results": results,
        "stage_b_gate": stage_b,
        "design_contract": {
            "complete_relation_model_trained_once_then_frozen": True,
            "scores_captured_from_trained_query_key_projections": True,
            "manual_qk_or_score_factorization": False,
            "labels_passed_to_intervention": False,
            "anchor_policy_uses_only_entity_masks": True,
            "train_valid_test_streams_distinct": True,
            "parameter_sweep": False,
            "walk_time": float(config["mechanism"]["walk_time"]),
            "query_scope": str(config["mechanism"].get("query_scope", "all")),
            "readout": str(config["mechanism"].get("readout", "row_transport")),
        },
        "limitations": [
            "This is a synthetic structural relation task, not a natural-language benchmark.",
            "A pass is seed-7 prequalification only and does not establish multi-seed stability.",
            "Exact matrix-exponential simulation does not establish hardware speedup.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "baseline_valid_accuracy": baseline["metrics"]["valid"]["accuracy"],
                "stage_a": stage_a,
                "stage_b": stage_b,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
