# Q-Attention

Q-Attention is a public research scaffold for developing **quantum-enhanced spectral key steering** methods for span-centric NLP information extraction.

The project is not a generic quantum-attention rewrite. Its core mechanism is spectral key steering:

```text
learn a task-specific key-space projector offline
        -> inject it into attention keys at inference time
        -> steer attention without changing model weights
```

The source code in this repository is written from scratch.

## Research Positioning

The planned method generalizes spectral key steering from controlled attention experiments to practical NLP tasks where models must focus on entity, event, aspect, or evidence spans.

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

## Current Status

```text
Stage: toy-data classical steering prototype
Code: tensor steering, encoder adapter, relation baseline, offline classical/quantum projector builders, spectral filter sweep, steered evaluator
Validation: baseline -> classical/quantum projector -> spectral filter sweep -> steering eval runs end to end on toy relation data
Visibility: public
```

This repository is not ready for formal large-scale GPU benchmarking yet. The next gate is adding real dataset adapters and configs.

## Project Docs

- Experiment runner guide: [docs/experiment_runner_guide.md](docs/experiment_runner_guide.md)
- Collaboration plan: [docs/collaboration_plan.md](docs/collaboration_plan.md)
- NLP task plan: [docs/nlp_task_plan.md](docs/nlp_task_plan.md)

## Project Structure

```text
src/q_attention/      # original implementation
experiments/          # training, projector-building, and evaluation scripts
examples/             # minimal demos and toy data
tests/                # unit tests
docs/                 # research notes and run guides
```

## Quick Check

Install the project in an active Python environment:

```bash
python -m pip install -e ".[dev]"
```

Run smoke tests:

```bash
python examples/minimal_key_steering.py
python examples/encoder_adapter_demo.py
python examples/quantum_projector_demo.py
python examples/spectral_filter_demo.py
python -m pytest -q
```

Run the current toy relation loop:

```bash
python experiments/train_relation_baseline.py --epochs 2 --batch_size 4 --output_dir runs/relation_toy --device cpu
python experiments/build_relation_projector.py --model_dir runs/relation_toy --batch_size 4 --device cpu --rank 4
python experiments/eval_relation_steering.py --model_dir runs/relation_toy --batch_size 4 --device cpu --gain 0.25 --output_dir runs/relation_toy/steering_eval
python experiments/build_relation_quantum_projector.py --model_dir runs/relation_toy --batch_size 4 --device cpu --rank 4 --num_qubits 4
python experiments/eval_relation_steering.py --model_dir runs/relation_toy --projector_path runs/relation_toy/relation_quantum_projector.pt --batch_size 4 --device cpu --gain 0.25 --output_dir runs/relation_toy/quantum_steering_eval
python experiments/sweep_relation_spectral_filters.py --model_dir runs/relation_toy --batch_size 4 --device cpu --families classical,quantum --modes hard_topk,high_pass,band_pass,soft_energy --ranks 2,4 --thresholds 0.5 --gains 0.25 --num_qubits 4 --output_dir runs/relation_toy/spectral_filter_sweep
python experiments/sweep_relation_spectral_filters.py --model_dir runs/relation_toy --batch_size 4 --device cpu --families classical,quantum --modes hard_topk,high_pass,band_pass,soft_energy --ranks 2,4 --gains 0.25 --num_qubits 4 --output_dir runs/relation_toy/spectral_filter_sweep
```

These commands are prototype checks, not paper-result benchmarks.
