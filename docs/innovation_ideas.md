# Innovation Ideas

## 1. Quantum-Kernel Adaptive Routing

Use a quantum feature map to compare a query representation with candidate steering subspaces:

```text
K_q(x, y) = |<Phi_q(x), Phi_q(y)>|^2
```

The router should remain lightweight, deterministic, and training-free in the first prototype.

## 2. Quantum-Inspired Spectral Projector Filtering

Use polynomial filters over singular values instead of hard truncation:

```text
P = U diag(f(S)) U^T
```

Candidate filters include high-pass, band-pass, and Chebyshev-style filters.

## 3. Quantum Separability-Guided Head and Layer Selection

Use a quantum-kernel separability score to decide which heads or layers should receive steering:

```text
score(layer, head) = MMD_q(positive_keys, negative_keys)
```

The aim is to reduce unnecessary interventions and improve interpretability.
