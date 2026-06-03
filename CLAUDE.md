# CLAUDE.md

## Project Goal

This repository is for an original quantum machine learning attention-steering project. The method should inherit the principle of spectral key steering while keeping all public source code clean-slate.

## Core Principle

The project should preserve this mechanism:

```text
learn projector P from key-space contrastive structure
apply k' = k + gPk inside attention layers
leave model weights unchanged
```

The project should not become a generic quantum-attention paper detached from this mechanism.

## Working Rules

- Do not copy external project source code into this repository.
- Write the implementation from scratch.
- Keep the public repository focused on original code, design notes, and application experiments.
- Use real application metrics, not only attention visualization.
- Keep first prototypes lightweight and torch-only unless a quantum framework is explicitly required.

## Research Themes

1. Quantum-kernel projector learning.
2. QSVT-inspired spectral projector filtering.
3. Quantum adaptive expert routing for key steering.
4. Practical deployment in attention-based time-series models.

## Preferred Application Track

Start with multivariate time-series anomaly detection or fault prediction because:

```text
attention is useful for long-range sensor dependencies
data is practical and application-facing
metrics are concrete
models are much cheaper than large language models
```

## First Implementation Target

Start with a minimal tensor-level key steering module under `src/q_attention/`:

```text
build_projector(...)
apply_key_steering(...)
quantum_kernel_projector(...)
```

Then add model adapters and experiments.
