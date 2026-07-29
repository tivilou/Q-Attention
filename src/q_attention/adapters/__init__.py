"""Adapters for injecting key steering into attention-based models."""

from .attention_scores import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from .encoder import EncoderKeySteeringAdapter, KeySteeringHookConfig, resolve_module
from .quantum_plugins import QuantumPluginHookConfig, QuantumPluginSteeringAdapter

__all__ = [
    "AttentionScoreHookConfig",
    "AttentionScoreKernelAdapter",
    "EncoderKeySteeringAdapter",
    "KeySteeringHookConfig",
    "QuantumPluginHookConfig",
    "QuantumPluginSteeringAdapter",
    "resolve_module",
]
