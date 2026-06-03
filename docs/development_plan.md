# Development Plan

## Goal

Build Q-Attention as a quantum-enhanced spectral key steering framework for practical attention-based models.

The project should inherit the key mechanism:

```text
learn P offline
apply k' = k + gPk at inference time
keep model weights frozen
```

## Phase 1: Minimal Tensor Prototype

Deliverables:

- `build_projector()` for spectral projector construction.
- `apply_key_steering()` for tensor-level key editing.
- Small examples showing `k' = k + gPk`.
- Unit tests for shape, stability, and unchanged unmasked positions.

Acceptance checks:

```text
python examples/minimal_demo.py
python -m pytest
```

## Phase 2: Quantum Projector Learning

Deliverables:

- Torch-only quantum-inspired feature map.
- Fidelity-style kernel.
- Projector construction from quantum-feature correlations.
- Linear-vs-quantum projector ablation.

Core form:

```text
K_q(x, y) = |<Phi_q(x), Phi_q(y)>|^2
```

## Phase 3: Spectral Filtering

Deliverables:

- Hard top-k projector.
- High-pass filter.
- Band-pass filter.
- Chebyshev-style filter.

Core form:

```text
P = U diag(f(S)) U^T
```

## Phase 4: Adaptive Expert Routing

Deliverables:

- Multiple projector experts.
- Query-dependent routing.
- Quantum-kernel expert scoring.
- Dynamic projector mixture.

Core form:

```text
P_dynamic = sum_m alpha_m P_m
```

## Phase 5: Attention-Model Adapter

Deliverables:

- Adapter for a small Transformer-style time-series model.
- Hook or module-level key editing.
- Frozen-backbone evaluation path.

The adapter should modify key representations, not model weights.

## Phase 6: Practical Application Experiment

Initial target:

```text
multivariate time-series anomaly detection / fault prediction
```

Compare:

```text
base attention model
linear spectral key steering
quantum projector learning
spectral filtering only
adaptive routing only
full Q-Attention
```

Report:

```text
F1
AUROC
AUPRC
false alarm rate
early warning time
latency
memory
projector norm stability
routing entropy
```

## Phase 7: Paper Assets

Deliverables:

- Method diagram.
- Ablation table.
- Runtime table.
- Sensitivity analysis.
- Attention/key-space visualization as secondary evidence.
