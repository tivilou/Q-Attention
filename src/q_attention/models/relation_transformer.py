"""A small Transformer-style relation extraction model with steerable keys."""

from __future__ import annotations

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
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self._split_heads(self.query_proj(hidden))
        key = self._split_heads(self.key_proj(hidden))
        value = self._split_heads(self.value_proj(hidden))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
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

    @property
    def key_module_paths(self) -> tuple[str, ...]:
        return tuple(f"encoder.layers.{idx}.attn.key_proj" for idx in range(self.config.num_layers))

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
        hidden = self.encoder(input_ids, attention_mask)
        subject_repr = self._masked_mean(hidden, subject_mask)
        object_repr = self._masked_mean(hidden, object_mask)
        context_repr = self._masked_mean(hidden, attention_mask)
        features = torch.cat([subject_repr, object_repr, context_repr], dim=-1)
        return self.classifier(features)
