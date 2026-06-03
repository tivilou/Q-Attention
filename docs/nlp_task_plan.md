# NLP Task Plan

## Unified Application

Q-Attention targets span-centric information extraction.

Unified input format:

```text
text
anchor spans
evidence spans or candidates
task type
```

Unified output format:

```text
structured label or role prediction
```

## Task 1: Relation Extraction

Purpose:

```text
verify the mechanism on entity-pair reasoning
```

Anchor:

```text
subject entity
object entity
```

Prediction:

```text
relation label
```

## Task 2: Event Argument Extraction

Purpose:

```text
test trigger-argument evidence steering
```

Anchor:

```text
event trigger
candidate argument span
```

Prediction:

```text
argument role label
```

## Task 3: Aspect-Based Sentiment Analysis

Purpose:

```text
test target-dependent opinion evidence steering
```

Anchor:

```text
aspect term
```

Prediction:

```text
sentiment label
```

## Task 4: Biomedical Relation Extraction

Purpose:

```text
test domain-specific low-resource information extraction
```

Anchor:

```text
biomedical entity pair
```

Prediction:

```text
biomedical relation label
```

## Cross-Task Experiments

Recommended experiments:

```text
single-task training and evaluation
multi-task projector bank
cross-task projector transfer
low-resource projector construction
robustness with distractor spans
```
