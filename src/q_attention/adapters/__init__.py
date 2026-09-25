"""Adapters for injecting attention interventions into research models."""

from .attention_context import AttentionContextHookConfig, AttentionContextKernelAdapter
from .attention_scores import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from .encoder import EncoderKeySteeringAdapter, KeySteeringHookConfig, resolve_module
from .quantum_plugins import QuantumPluginHookConfig, QuantumPluginSteeringAdapter

__all__ = [
    "AttentionContextHookConfig",
    "AttentionContextKernelAdapter",
    "AttentionScoreHookConfig",
    "AttentionScoreKernelAdapter",
    "EncoderKeySteeringAdapter",
    "KeySteeringHookConfig",
    "QuantumPluginHookConfig",
    "QuantumPluginSteeringAdapter",
    "resolve_module",
]
