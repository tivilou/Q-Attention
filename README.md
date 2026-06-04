# Q-Attention

Q-Attention is a public research scaffold for developing **quantum-enhanced spectral key steering** methods for span-centric NLP information extraction.

The project is not a generic quantum-attention rewrite. Its core mechanism is inherited from spectral key steering:

```text
learn a task-specific key-space projector offline
        -> inject it into attention keys at inference time
        -> steer attention without changing model weights
```

The source code in this repository will be written from scratch. Reference implementations may be studied locally, but third-party source code should not be copied into this repository.

## Research Positioning

The planned method generalizes spectral key steering from prompt-focused experiments to practical NLP tasks where models must focus on entity, event, aspect, or evidence spans.

Target task family:

```text
Span-Centric Information Extraction
```

Representative tasks:

```text
Relation Extraction
Event Argument Extraction
Aspect-Based Sentiment Analysis
Biomedical Relation Extraction
```

These tasks share a common structure:

```text
text + anchor spans + evidence spans -> structured prediction
```

## Core Mechanism

The basic intervention is:

```text
k' = k + gPk
```

where:

```text
k  = key representation inside an attention layer
P  = task-specific spectral projector
g  = steering strength
k' = steered key representation
```

For NLP information extraction, the steered keys usually correspond to:

```text
entity spans
relation evidence spans
event triggers
candidate arguments
aspect terms
opinion clues
biomedical entity mentions
```

## Planned Contributions

```text
1. Quantum-kernel projector learning for span-evidence relevance
2. QSVT-inspired spectral projector filtering
3. Quantum adaptive expert routing across related NLP tasks
4. Multi-task evaluation on span-centric information extraction
```

## Project Docs

- Experiment runner guide: [docs/experiment_runner_guide.md](docs/experiment_runner_guide.md)
- Collaboration plan: [docs/collaboration_plan.md](docs/collaboration_plan.md)
- NLP task plan: [docs/nlp_task_plan.md](docs/nlp_task_plan.md)

## Current Status

```text
Stage: research design
Code: relation extraction baseline scaffolded
Visibility: public
```

## Planned Structure

```text
src/q_attention/      # original implementation
experiments/          # benchmark and ablation scripts
docs/                 # research notes and design records
```

## Current Baseline Command

```bash
python experiments/train_relation_baseline.py --epochs 5 --output_dir runs/relation_toy
```

## First Milestone

Run the minimal tensor demo and encoder adapter demo, then add the first relation extraction baseline before extending to event, aspect, and biomedical extraction.
