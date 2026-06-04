# Development Plan

## Goal

Build Q-Attention as a quantum-enhanced spectral key steering framework for span-centric NLP information extraction.

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
python examples/minimal_key_steering.py
python examples/encoder_adapter_demo.py
python -m pytest -q
```

## Phase 2: NLP Encoder Adapter

Deliverables:

- Adapter for Transformer encoder attention layers.
- Span mask utilities for entity, trigger, argument, and aspect spans.
- Frozen-backbone key steering path.

The adapter should modify key representations, not model weights.

## Phase 3: First Task - Relation Extraction

Deliverables:

- Relation extraction data adapter.
- Entity-pair anchor representation.
- Baseline encoder model.
- Offline anchor-key projector builder.
- Frozen-backbone key-steered evaluator.
- Later extension: positive/negative evidence-aware projector construction.

Core form:

```text
text + entity pair -> relation label
```

## Phase 4: Quantum Projector Learning

Deliverables:

- Torch-only quantum-inspired feature map.
- Fidelity-style kernel.
- Projector construction from quantum-feature correlations.
- Linear-vs-quantum projector ablation.

Core form:

```text
K_q(x, y) = |<Phi_q(x), Phi_q(y)>|^2
```

Toy acceptance checks:

```text
python examples/quantum_projector_demo.py
python experiments/build_relation_quantum_projector.py --model_dir runs/relation_toy --batch_size 4 --device cpu --rank 4 --num_qubits 4
```

## Phase 5: Spectral Filtering

Deliverables:

- Hard top-k projector.
- High-pass filter.
- Band-pass filter.
- Chebyshev-style filter.

Core form:

```text
P = U diag(f(S)) U^T
```

Toy acceptance checks:

```text
python examples/spectral_filter_demo.py
python experiments/sweep_relation_spectral_filters.py --model_dir runs/relation_toy --batch_size 4 --device cpu --families classical,quantum --modes hard_topk,high_pass,band_pass,soft_energy --ranks 2,4 --gains 0.25
```

## Phase 6: Multi-Task Expansion

Add related NLP tasks:

```text
Event Argument Extraction
Aspect-Based Sentiment Analysis
Biomedical Relation Extraction
```

Each task should define:

```text
anchor spans
evidence spans
positive/negative projector-building examples
structured prediction metric
```

## Phase 7: Adaptive Expert Routing

Deliverables:

- Multiple projector experts.
- Query-dependent routing.
- Quantum-kernel expert scoring.
- Dynamic projector mixture.

Core form:

```text
P_dynamic = sum_m alpha_m P_m
```

Toy acceptance checks:

```text
python examples/routing_demo.py
python experiments/eval_relation_routing.py --model_dir runs/relation_toy --batch_size 4 --device cpu --gain 0.25 --temperature 0.5 --rank 2 --num_qubits 4
```

## Phase 8: Experimental Suite

Compare:

```text
base encoder
linear spectral key steering
quantum projector learning
spectral filtering only
adaptive routing only
full Q-Attention
```

Report:

```text
Micro-F1
Macro-F1
Precision
Recall
Accuracy where appropriate
low-resource performance
robustness under distractors
latency
memory
projector norm stability
routing entropy
```

## Phase 9: Paper Assets

Deliverables:

- Method diagram.
- Algorithm pseudocode.
- Multi-task results table.
- Ablation table.
- Low-resource table.
- Robustness table.
- Runtime table.
- Sensitivity analysis.
- Key-space visualization as secondary evidence.
