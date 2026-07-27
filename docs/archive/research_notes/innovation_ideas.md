# Innovation Ideas

## Research Frame

The project should remain grounded in spectral key steering:

```text
offline key-space projector learning
runtime key editing
adaptive routing across projector experts
```

The application frame is span-centric NLP information extraction rather than a single relation extraction task.

## Unified NLP Problem

Many practical NLP tasks share the same structure:

```text
text + anchor spans + evidence spans -> structured output
```

Examples:

```text
Relation Extraction: entity pair -> relation label
Event Argument Extraction: trigger + argument candidate -> role label
Aspect-Based Sentiment Analysis: aspect term -> sentiment label
Biomedical Relation Extraction: biomedical entity pair -> relation label
```

The method should steer attention toward the span-level evidence needed for the prediction.

## 1. Quantum-Kernel Projector Learning

Instead of learning the projector only from linear key-space covariance, map key representations through a quantum-inspired feature map:

```text
Phi_q(k)
```

Then build projectors from quantum-feature correlations:

```text
Omega_q = Phi_q(H)^T Phi_q(H_plus)
P_q = spectral_projector(Omega_q)
```

Expected benefit:

```text
capture nonlinear span-evidence relevance structure in key space
```

## 2. QSVT-Inspired Spectral Projector Filtering

Replace hard singular-vector truncation with smooth spectral filtering:

```text
P = U diag(f(S)) U^T
```

Candidate filters:

```text
high-pass relevance filter
band-pass noise suppression filter
Chebyshev-style spectral filter
positive/negative contrast filter
```

Expected benefit:

```text
more stable projector construction than hard top-k selection
```

## 3. Quantum Adaptive Expert Routing

Build projector experts for related NLP tasks or evidence types:

```text
entity-pair expert
trigger-argument expert
aspect-opinion expert
biomedical-entity expert
```

Use a quantum-kernel routing score:

```text
score_m = sum_j K_q(query, U_m[:, j]) * weight_j
```

Then mix experts:

```text
P_dynamic = sum_m alpha_m P_m
```

Expected benefit:

```text
input-aware key steering across related NLP information extraction tasks
```

## 4. Multi-Task Application Track

Do not evaluate only whether attention moves. Evaluate practical NLP task performance.

Recommended tasks:

```text
Relation Extraction
Event Argument Extraction
Aspect-Based Sentiment Analysis
Biomedical Relation Extraction
```

Recommended metrics:

```text
Micro-F1
Macro-F1
Precision
Recall
Accuracy where appropriate
Robustness under distractors
Low-resource performance
Latency
Memory
```

## Recommended Paper Claim

A concise claim:

> We propose a quantum-enhanced spectral key steering framework for span-centric information extraction, learning task-specific key-space projectors and injecting them into attention layers at inference time to improve structured NLP prediction without updating model weights.
