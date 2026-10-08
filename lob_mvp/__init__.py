"""Reproducible MVP for regime-aware order-flow representation learning."""

from .core import MVPConfig, build_features, generate_synthetic_events

__all__ = ["MVPConfig", "build_features", "generate_synthetic_events"]
