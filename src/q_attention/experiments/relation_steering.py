"""Shared utilities for relation extraction steering experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from q_attention.adapters import EncoderKeySteeringAdapter, KeySteeringHookConfig, resolve_module
from q_attention.metrics import classification_metrics
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.projectors import SpectralProjectorConfig, build_projector
from q_attention.tasks.relation import PAD_TOKEN, RelationDataset, RelationRecord, collate_relation_batch

ANCHOR_CHOICES = ("subject", "object", "subject_object", "all_tokens")


@dataclass(frozen=True)
class RelationRunArtifacts:
    """Loaded baseline checkpoint plus metadata needed by steering scripts."""

    model_dir: Path
    model: RelationExtractionModel
    vocab: dict[str, int]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    key_module_paths: tuple[str, ...]
    args: Mapping[str, Any]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class KeyCollection:
    """Anchor key vectors collected from one or more key projection layers."""

    keys: torch.Tensor
    layer_counts: dict[str, int]
    num_batches: int
    sampled_from: int | None = None


@dataclass(frozen=True)
class RelationLayerSamples:
    """Label-aligned key samples collected from one steerable layer."""

    keys: torch.Tensor
    relation_features: torch.Tensor
    labels: torch.Tensor
    sampled_from: int | None = None


@dataclass(frozen=True)
class RelationKeyCollection:
    """Per-example relation features aligned with steerable key vectors."""

    keys: torch.Tensor
    relation_features: torch.Tensor
    labels: torch.Tensor
    layer_counts: dict[str, int]
    num_batches: int
    sampled_from: int | None = None
    layer_samples: dict[str, RelationLayerSamples] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Classification metrics and per-example outputs."""

    metrics: dict[str, float]
    predictions: list[int]
    labels: list[int]


def choose_device(name: str) -> torch.device:
    """Resolve an experiment device name."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a collated relation batch to a device."""
    return {key: value.to(device) for key, value in batch.items()}


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))




def torch_load_weights(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load trusted project tensors while avoiding pickle warnings on newer PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        return torch.load(path, map_location=map_location)

def load_relation_run(model_dir: str | Path, device: torch.device) -> RelationRunArtifacts:
    """Load a baseline relation checkpoint and reconstruct its model."""
    model_dir = Path(model_dir)
    metrics_path = model_dir / "metrics.json"
    model_path = model_dir / "model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing baseline metrics: {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing baseline checkpoint: {model_path}")

    metrics = read_json(metrics_path)
    args = metrics.get("args", {})
    vocab_path = model_dir / "vocab.json"
    labels_path = model_dir / "labels.json"
    vocab = read_json(vocab_path) if vocab_path.exists() else metrics["vocab"]
    label_to_id = read_json(labels_path) if labels_path.exists() else metrics["label_to_id"]
    label_to_id = {str(label): int(idx) for label, idx in label_to_id.items()}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    state_dict = torch_load_weights(model_path, map_location="cpu")
    max_length = int(state_dict["encoder.position_embedding.weight"].shape[0])
    dim = int(args.get("dim", state_dict["encoder.token_embedding.weight"].shape[1]))
    key_paths = tuple(metrics.get("key_module_paths") or ())
    config = RelationTransformerConfig(
        vocab_size=len(vocab),
        num_labels=len(label_to_id),
        dim=dim,
        num_layers=int(args.get("num_layers", len(key_paths) or 1)),
        num_heads=int(args.get("num_heads", 4)),
        ff_dim=int(args.get("ff_dim", dim * 2)),
        dropout=float(args.get("dropout", 0.0)),
        max_length=max_length,
    )
    model = RelationExtractionModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    key_module_paths = key_paths or model.key_module_paths
    return RelationRunArtifacts(
        model_dir=model_dir,
        model=model,
        vocab={str(token): int(idx) for token, idx in vocab.items()},
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        key_module_paths=key_module_paths,
        args=args,
        metrics=metrics,
    )


def make_relation_loader(
    records: Sequence[RelationRecord],
    vocab: Mapping[str, int],
    label_to_id: Mapping[str, int],
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader with the project relation collator."""
    dataset = RelationDataset(list(records), vocab, label_to_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_relation_batch(batch, pad_id=vocab[PAD_TOKEN]),
    )


def anchor_mask_from_batch(batch: Mapping[str, torch.Tensor], anchor: str = "subject_object") -> torch.Tensor:
    """Build the token mask used for projector learning or inference steering."""
    if anchor == "subject":
        return batch["subject_mask"].to(dtype=torch.bool)
    if anchor == "object":
        return batch["object_mask"].to(dtype=torch.bool)
    if anchor == "subject_object":
        return (batch["subject_mask"] | batch["object_mask"]).to(dtype=torch.bool)
    if anchor == "all_tokens":
        return batch["attention_mask"].to(dtype=torch.bool)
    raise ValueError(f"unknown anchor '{anchor}', expected one of {ANCHOR_CHOICES}")


def _tensor_from_hook_output(output: object, path: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"key module '{path}' must return a tensor or a tuple/list whose first item is a tensor")


def _sample_keys(keys: torch.Tensor, max_vectors: int | None, seed: int) -> tuple[torch.Tensor, int | None]:
    if max_vectors is None or max_vectors <= 0 or keys.shape[0] <= max_vectors:
        return keys, None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(keys.shape[0], generator=generator)[:max_vectors]
    return keys[indices], int(keys.shape[0])


def collect_anchor_key_vectors(
    model: RelationExtractionModel,
    loader: DataLoader,
    device: torch.device,
    key_module_paths: Sequence[str],
    *,
    anchor: str = "subject_object",
    max_vectors: int | None = None,
    seed: int = 13,
) -> KeyCollection:
    """Collect key-projection outputs at anchor-token positions."""
    if not key_module_paths:
        raise ValueError("at least one key module path is required")

    captured: dict[str, torch.Tensor] = {}
    layer_counts = {path: 0 for path in key_module_paths}
    vectors: list[torch.Tensor] = []
    handles = []
    num_batches = 0

    def make_hook(path: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            captured[path] = _tensor_from_hook_output(output, path).detach().cpu()

        return hook

    try:
        for path in key_module_paths:
            handles.append(resolve_module(model, path).register_forward_hook(make_hook(path)))

        model.eval()
        with torch.no_grad():
            for batch in loader:
                num_batches += 1
                batch = move_batch(batch, device)
                captured.clear()
                _ = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
                mask = anchor_mask_from_batch(batch, anchor).detach().cpu()

                for path in key_module_paths:
                    if path not in captured:
                        raise RuntimeError(f"hook for '{path}' did not capture any tensor")
                    keys = captured[path]
                    if keys.shape[:-1] != mask.shape:
                        raise ValueError(f"captured keys {keys.shape} do not match anchor mask {mask.shape}")
                    selected = keys[mask]
                    if selected.numel() > 0:
                        vectors.append(selected.float())
                        layer_counts[path] += int(selected.shape[0])
    finally:
        for handle in handles:
            handle.remove()

    if not vectors:
        raise ValueError("no anchor key vectors were collected")

    keys = torch.cat(vectors, dim=0)
    keys, sampled_from = _sample_keys(keys, max_vectors=max_vectors, seed=seed)
    return KeyCollection(keys=keys, layer_counts=layer_counts, num_batches=num_batches, sampled_from=sampled_from)


def relation_pair_features(
    keys: torch.Tensor,
    subject_mask: torch.Tensor,
    object_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one key vector and a subject-object feature per example."""
    if keys.ndim != 3:
        raise ValueError("keys must have shape (batch, sequence, dim)")
    if subject_mask.shape != keys.shape[:2] or object_mask.shape != keys.shape[:2]:
        raise ValueError("subject/object masks must match key batch and sequence dimensions")
    subject_mask = subject_mask.to(device=keys.device, dtype=keys.dtype)
    object_mask = object_mask.to(device=keys.device, dtype=keys.dtype)
    subject_denominator = subject_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    object_denominator = object_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    subject = torch.sum(keys * subject_mask.unsqueeze(-1), dim=1) / subject_denominator
    object_ = torch.sum(keys * object_mask.unsqueeze(-1), dim=1) / object_denominator
    pair_features = torch.cat((subject, object_, subject - object_, subject * object_), dim=-1)
    key_vectors = 0.5 * (subject + object_)
    return key_vectors.float(), pair_features.float()


def collect_relation_key_samples(
    model: RelationExtractionModel,
    loader: DataLoader,
    device: torch.device,
    key_module_paths: Sequence[str],
    *,
    max_vectors: int | None = None,
    seed: int = 13,
    collect_by_layer: bool = False,
    max_vectors_per_layer: int | None = None,
) -> RelationKeyCollection:
    """Collect label-aligned relation features from every steerable key layer."""
    if not key_module_paths:
        raise ValueError("at least one key module path is required")

    captured: dict[str, torch.Tensor] = {}
    layer_counts = {path: 0 for path in key_module_paths}
    key_vectors: list[torch.Tensor] = []
    pair_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    layer_key_vectors: dict[str, list[torch.Tensor]] = {path: [] for path in key_module_paths}
    layer_pair_features: dict[str, list[torch.Tensor]] = {path: [] for path in key_module_paths}
    layer_labels: dict[str, list[torch.Tensor]] = {path: [] for path in key_module_paths}
    handles = []
    num_batches = 0

    def make_hook(path: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            captured[path] = _tensor_from_hook_output(output, path).detach().cpu()

        return hook

    try:
        for path in key_module_paths:
            handles.append(resolve_module(model, path).register_forward_hook(make_hook(path)))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                num_batches += 1
                batch = move_batch(batch, device)
                captured.clear()
                _ = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
                subject_mask = batch["subject_mask"].detach().cpu()
                object_mask = batch["object_mask"].detach().cpu()
                batch_labels = batch["labels"].detach().cpu().long()
                for path in key_module_paths:
                    if path not in captured:
                        raise RuntimeError(f"hook for '{path}' did not capture any tensor")
                    keys = captured[path]
                    if keys.shape[:-1] != subject_mask.shape:
                        raise ValueError(f"captured keys {keys.shape} do not match relation masks {subject_mask.shape}")
                    relation_keys, relation_z = relation_pair_features(keys, subject_mask, object_mask)
                    key_vectors.append(relation_keys)
                    pair_features.append(relation_z)
                    labels.append(batch_labels)
                    if collect_by_layer:
                        layer_key_vectors[path].append(relation_keys)
                        layer_pair_features[path].append(relation_z)
                        layer_labels[path].append(batch_labels)
                    layer_counts[path] += int(batch_labels.shape[0])
    finally:
        for handle in handles:
            handle.remove()

    if not key_vectors:
        raise ValueError("no relation key samples were collected")
    keys = torch.cat(key_vectors, dim=0)
    features = torch.cat(pair_features, dim=0)
    target = torch.cat(labels, dim=0)
    if max_vectors is not None and max_vectors > 0 and keys.shape[0] > max_vectors:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(keys.shape[0], generator=generator)[:max_vectors]
        sampled_from: int | None = int(keys.shape[0])
        keys, features, target = keys[indices], features[indices], target[indices]
    else:
        sampled_from = None
    per_layer: dict[str, RelationLayerSamples] = {}
    if collect_by_layer:
        for layer_index, path in enumerate(key_module_paths):
            layer_keys = torch.cat(layer_key_vectors[path], dim=0)
            layer_features = torch.cat(layer_pair_features[path], dim=0)
            layer_target = torch.cat(layer_labels[path], dim=0)
            layer_sampled_from: int | None = None
            if max_vectors_per_layer is not None and max_vectors_per_layer > 0 and layer_keys.shape[0] > max_vectors_per_layer:
                generator = torch.Generator(device="cpu").manual_seed(seed + layer_index)
                indices = torch.randperm(layer_keys.shape[0], generator=generator)[:max_vectors_per_layer]
                layer_sampled_from = int(layer_keys.shape[0])
                layer_keys = layer_keys[indices]
                layer_features = layer_features[indices]
                layer_target = layer_target[indices]
            per_layer[path] = RelationLayerSamples(
                keys=layer_keys,
                relation_features=layer_features,
                labels=layer_target,
                sampled_from=layer_sampled_from,
            )
    return RelationKeyCollection(
        keys=keys,
        relation_features=features,
        labels=target,
        layer_counts=layer_counts,
        num_batches=num_batches,
        sampled_from=sampled_from,
        layer_samples=per_layer,
    )


def build_anchor_projector(
    keys: torch.Tensor,
    config: SpectralProjectorConfig,
    *,
    center: bool = False,
) -> torch.Tensor:
    """Build a spectral projector from collected anchor keys."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape (num_vectors, dim)")
    if keys.shape[0] == 0:
        raise ValueError("at least one key vector is required")
    if center:
        keys = keys - keys.mean(dim=0, keepdim=True)
    return build_projector(keys, keys, config=config)


def load_projector(
    path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], Mapping[str, Any]]:
    """Load a shared or layer-specific projector payload."""
    payload = torch_load_weights(Path(path), map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.to(device), {}
    if isinstance(payload, Mapping) and "projector" in payload:
        projector = payload["projector"]
        if not isinstance(projector, torch.Tensor):
            raise TypeError("projector payload field must be a tensor")
        metadata = payload.get("metadata", {})
        return projector.to(device), metadata if isinstance(metadata, Mapping) else {}
    if isinstance(payload, Mapping) and "projectors" in payload:
        raw_projectors = payload["projectors"]
        if not isinstance(raw_projectors, Mapping) or not raw_projectors:
            raise TypeError("projectors payload field must be a non-empty mapping")
        projectors: dict[str, torch.Tensor] = {}
        for module_path, projector in raw_projectors.items():
            if not isinstance(module_path, str) or not isinstance(projector, torch.Tensor):
                raise TypeError("projectors must map module-path strings to tensors")
            projectors[module_path] = projector.to(device)
        metadata = payload.get("metadata", {})
        return projectors, metadata if isinstance(metadata, Mapping) else {}
    raise ValueError(f"unsupported projector payload in {path}")


def projector_shape_summary(
    projector: torch.Tensor | Mapping[str, torch.Tensor],
) -> list[int] | dict[str, list[int]]:
    """Return JSON-safe shapes for shared or layer-specific projectors."""
    if isinstance(projector, torch.Tensor):
        return list(projector.shape)
    return {path: list(layer_projector.shape) for path, layer_projector in projector.items()}


def evaluate_relation_model(
    model: RelationExtractionModel,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    *,
    projector: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    key_module_paths: Sequence[str] = (),
    gain: float = 1.0,
    anchor: str = "subject_object",
) -> EvaluationResult:
    """Evaluate the relation model, optionally with frozen-backbone key steering."""
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    adapter = EncoderKeySteeringAdapter(model, key_module_paths) if projector is not None else None

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if adapter is None:
                logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
            else:
                mask = anchor_mask_from_batch(batch, anchor)
                hook_config = KeySteeringHookConfig(projector=projector, mask=mask, gain=gain)
                with adapter.steering(hook_config):
                    logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])

            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            total_items += int(batch["labels"].shape[0])
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return EvaluationResult(metrics=metrics, predictions=predictions, labels=labels)
