"""Adapters for injecting key steering into attention-based models."""

from .encoder import EncoderKeySteeringAdapter, KeySteeringHookConfig, resolve_module
from .quantum_plugins import QuantumPluginHookConfig, QuantumPluginSteeringAdapter

__all__ = [
    "EncoderKeySteeringAdapter",
    "KeySteeringHookConfig",
    "QuantumPluginHookConfig",
    "QuantumPluginSteeringAdapter",
    "resolve_module",
]
