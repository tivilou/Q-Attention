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
Stage: real-data pre-handoff scaffold
Code: tensor steering, encoder adapter, relation baseline, real-data converters, smoke pipeline, offline classical/quantum projector builders, spectral filter sweep, adaptive routing, steered evaluator
Validation: toy loop passes; real-data converter and pipeline entry points are being validated before GPU handoff
Visibility: public
```

This repository is not ready for formal large-scale GPU benchmarking yet. The next gate is completing a small selected-real-dataset smoke run and then freezing a handoff command for dzy958.

## Project Docs

- Experiment runner guide: [docs/experiment_runner_guide.md](docs/experiment_runner_guide.md)
- Real-data smoke plan: [docs/real_data_smoke_plan.md](docs/real_data_smoke_plan.md)
- Re-TACRED experiment handoff: [docs/retacred_experiment_handoff.md](docs/retacred_experiment_handoff.md)
- Collaboration plan: [docs/collaboration_plan.md](docs/collaboration_plan.md)
- NLP task plan: [docs/nlp_task_plan.md](docs/nlp_task_plan.md)

## Project Structure

```text
src/q_attention/      # original implementation
experiments/          # training, data preparation, smoke pipeline, projector-building, and evaluation scripts
configs/              # experiment and smoke-run config templates
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
python examples/routing_demo.py
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
python experiments/eval_relation_routing.py --model_dir runs/relation_toy --batch_size 4 --device cpu --gain 0.25 --temperature 0.5 --rank 2 --num_qubits 4 --output_dir runs/relation_toy/relation_routing_eval
```

These commands are prototype checks, not paper-result benchmarks.

Prepare canonical real relation data:

```bash
python experiments/prepare_relation_data.py --format tacred_json --dataset_name retacred --train_path data/raw/retacred/train.json --valid_path data/raw/retacred/dev.json --test_path data/raw/retacred/test.json --output_dir data/relation/retacred
```

Run a tiny real-data smoke chain after conversion:

```bash
python experiments/run_relation_smoke_pipeline.py --config data/relation/retacred/data_config.json --output_dir runs/retacred_real_smoke --device cpu --max_train_records 256 --max_valid_records 128
```