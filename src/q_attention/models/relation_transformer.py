"""A small Transformer-style relation extraction model with steerable keys."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RelationTransformerConfig:
    vocab_size: int
    num_labels: int
    dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    max_length: int = 256


class AttentionScorePassThrough(nn.Module):
    """Explicit hook point for behavior-preserving score interventions."""

    def forward(
        self,
        scores: torch.Tensor,
        _query: torch.Tensor,
        _key: torch.Tensor,
        _value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return scores


class SteerableSelfAttention(nn.Module):
    """Self-attention layer with an explicit key projection module."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.score_intervention = AttentionScorePassThrough()
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self._split_heads(self.query_proj(hidden))
        key = self._split_heads(self.key_proj(hidden))
        value = self._split_heads(self.value_proj(hidden))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = self.score_intervention(scores, query, key, value)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().view(hidden.shape)
        return self.out_proj(context)


class SteerableEncoderLayer(nn.Module):
    def __init__(self, config: RelationTransformerConfig) -> None:
        super().__init__()
        self.attn = SteerableSelfAttention(config.dim, config.num_heads, config.dropout)
        self.attn_norm = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.dim),
        )
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.attn_norm(hidden + self.dropout(self.attn(hidden, attention_mask)))
        hidden = self.ffn_norm(hidden + self.dropout(self.ffn(hidden)))
        return hidden


class SteerableEncoder(nn.Module):
    def __init__(self, config: RelationTransformerConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.layers = nn.ModuleList([SteerableEncoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens = input_ids.shape
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0).expand(batch, tokens)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class RelationExtractionModel(nn.Module):
    """Relation classifier using subject/object span pooling."""

    def __init__(self, config: RelationTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SteerableEncoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.dim * 3, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.num_labels),
        )
        self._model_parallel_devices: tuple[torch.device, ...] = ()
        self._model_parallel_layer_devices: tuple[torch.device, ...] = ()

    @property
    def model_parallel_enabled(self) -> bool:
        """Whether this model is configured with explicit layer sharding."""
        return bool(self._model_parallel_devices)

    @property
    def model_parallel_input_device(self) -> torch.device:
        """Device receiving token IDs and the first encoder stage."""
        if not self.model_parallel_enabled:
            return next(self.parameters()).device
        return self._model_parallel_devices[0]

    @property
    def model_parallel_output_device(self) -> torch.device:
        """Device hosting the classifier and returned logits."""
        if not self.model_parallel_enabled:
            return next(self.parameters()).device
        return self._model_parallel_devices[-1]

    @property
    def model_parallel_layer_devices(self) -> tuple[torch.device, ...]:
        """Device for each encoder layer, in layer order."""
        return self._model_parallel_layer_devices

    def configure_model_parallel(
        self, devices: Sequence[torch.device | str]
    ) -> "RelationExtractionModel":
        """Shard complete encoder layers across explicit devices.

        This is layer/pipeline parallelism, not tensor parallelism: each layer
        remains intact and hidden states are transferred between stages.
        """
        normalized = tuple(torch.device(device) for device in devices)
        if len(normalized) < 2:
            raise ValueError("model parallelism requires at least two devices")
        if len(normalized) > self.config.num_layers:
            raise ValueError(
                "model parallel device count cannot exceed the number of encoder layers"
            )
        if any(device.type not in {"cpu", "cuda"} for device in normalized):
            raise ValueError("model parallel devices must be CPU or CUDA devices")
        if any(device.type == "cuda" and device.index is None for device in normalized):
            raise ValueError("model parallel CUDA devices must include explicit indexes")
        if any(device.type == "cuda" and not torch.cuda.is_available() for device in normalized):
            raise RuntimeError("CUDA model parallelism requested but CUDA is unavailable")
        cuda_indices = [device.index for device in normalized if device.type == "cuda"]
        if len(cuda_indices) != len(set(cuda_indices)):
            raise ValueError("model parallel CUDA devices must be unique")

        layer_devices = tuple(
            normalized[min(index * len(normalized) // self.config.num_layers, len(normalized) - 1)]
            for index in range(self.config.num_layers)
        )
        self.encoder.token_embedding.to(normalized[0])
        self.encoder.position_embedding.to(normalized[0])
        self.encoder.dropout.to(normalized[0])
        for layer, device in zip(self.encoder.layers, layer_devices):
            layer.to(device)
        self.classifier.to(normalized[-1])
        self._model_parallel_devices = normalized
        self._model_parallel_layer_devices = layer_devices
        return self

    def model_parallel_metadata(self) -> dict[str, object]:
        """Return a JSON-safe module/device map for run provenance."""
        if not self.model_parallel_enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "devices": [str(device) for device in self._model_parallel_devices],
            "module_devices": {
                "encoder.token_embedding": str(self._model_parallel_devices[0]),
                "encoder.position_embedding": str(self._model_parallel_devices[0]),
                **{
                    f"encoder.layers.{index}": str(device)
                    for index, device in enumerate(self._model_parallel_layer_devices)
                },
                "classifier": str(self._model_parallel_devices[-1]),
            },
        }

    @property
    def key_module_paths(self) -> tuple[str, ...]:
        return tuple(f"encoder.layers.{idx}.attn.key_proj" for idx in range(self.config.num_layers))

    @property
    def score_module_paths(self) -> tuple[str, ...]:
        return tuple(
            f"encoder.layers.{idx}.attn.score_intervention"
            for idx in range(self.config.num_layers)
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=hidden.device, dtype=hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(hidden * mask.unsqueeze(-1), dim=1) / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model_parallel_enabled:
            hidden = self.encoder(input_ids, attention_mask)
        else:
            input_device = self._model_parallel_devices[0]
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)
            batch, tokens = input_ids.shape
            positions = torch.arange(tokens, device=input_device).unsqueeze(0).expand(batch, tokens)
            hidden = self.encoder.token_embedding(input_ids) + self.encoder.position_embedding(positions)
            hidden = self.encoder.dropout(hidden)
            for layer, layer_device in zip(self.encoder.layers, self._model_parallel_layer_devices):
                hidden = hidden.to(layer_device)
                hidden = layer(hidden, attention_mask.to(layer_device))
        subject_repr = self._masked_mean(hidden, subject_mask)
        object_repr = self._masked_mean(hidden, object_mask)
        context_repr = self._masked_mean(hidden, attention_mask)
        features = torch.cat([subject_repr, object_repr, context_repr], dim=-1)
        return self.classifier(features)
