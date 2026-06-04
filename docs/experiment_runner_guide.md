# Experiment Runner Guide

This guide is for `dzy958`, who is responsible for running experiments, collecting logs, and reporting results.

## Role

The experiment runner should not modify core source code during a run. The expected workflow is:

```text
pull latest main
activate your own experiment environment
install the project in editable mode
run the provided command
save logs and metrics
report success or failure with exact details
```

## Repository

```text
https://github.com/tivilou/Q-Attention
```

## Environment Policy

Do not assume that your environment matches the coding server environment.

Use your own conda or virtualenv environment. After activation, all commands should use the environment-local `python`.

Recommended setup:

```bash
conda create -n q-attention python=3.10 -y
conda activate q-attention
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you already have a suitable environment, activate it and run:

```bash
python -m pip install -e ".[dev]"
```

Record your environment:

```bash
which python
python --version
python -c "import torch; print(torch.__version__)"
python -m pytest --version
```

## Pull Latest Code

```bash
git clone https://github.com/tivilou/Q-Attention.git
cd Q-Attention
```

If you already cloned it:

```bash
cd Q-Attention
git pull
```

Record the commit before running anything:

```bash
git rev-parse HEAD
```

## Current Smoke Tests

Run these after every pull:

```bash
python examples/minimal_key_steering.py
python examples/encoder_adapter_demo.py
python examples/quantum_projector_demo.py
python examples/spectral_filter_demo.py
python examples/routing_demo.py
python -m pytest -q
```

Expected result:

```text
minimal_key_steering demo passes
encoder_adapter demo passes
quantum_projector demo passes
spectral_filter demo passes
routing demo passes
pytest reports all tests passed
```

Current expected pytest count:

```text
33 passed
```

## Current Relation Baseline Dry Run

After smoke tests pass, run the toy relation extraction baseline:

```bash
python experiments/train_relation_baseline.py --epochs 2 --batch_size 4 --output_dir runs/relation_toy --device cpu
```

Expected outputs:

```text
runs/relation_toy/model.pt
runs/relation_toy/metrics.json
runs/relation_toy/vocab.json
runs/relation_toy/labels.json
```

This is not a formal benchmark. It only verifies that the relation extraction training/evaluation pipeline works end to end.

## Current Classical Steering Dry Run

After the baseline dry run succeeds, build the offline spectral projector from anchor-span keys:

```bash
python experiments/build_relation_projector.py --model_dir runs/relation_toy --batch_size 4 --device cpu --rank 4
```

Expected outputs:

```text
runs/relation_toy/relation_projector.pt
runs/relation_toy/relation_projector_metadata.json
```

Then evaluate frozen-backbone key steering:

```bash
python experiments/eval_relation_steering.py --model_dir runs/relation_toy --batch_size 4 --device cpu --gain 0.25 --output_dir runs/relation_toy/steering_eval
```

Expected outputs:

```text
runs/relation_toy/steering_eval/metrics.json
runs/relation_toy/steering_eval/predictions.jsonl
runs/relation_toy/steering_eval/run_info.json
```

This is still a toy-data prototype check. Do not treat these numbers as paper results.

## Current Toy Quantum Projector Dry Run

After the baseline dry run succeeds, build the toy quantum-inspired projector:

```bash
python experiments/build_relation_quantum_projector.py --model_dir runs/relation_toy --batch_size 4 --device cpu --rank 4 --num_qubits 4 --angle_scale 1.25
```

Expected outputs:

```text
runs/relation_toy/relation_quantum_projector.pt
runs/relation_toy/relation_quantum_projector_metadata.json
```

Then evaluate the same frozen-backbone steering path with the quantum projector:

```bash
python experiments/eval_relation_steering.py --model_dir runs/relation_toy --projector_path runs/relation_toy/relation_quantum_projector.pt --batch_size 4 --device cpu --gain 0.25 --output_dir runs/relation_toy/quantum_steering_eval
```

Expected outputs:

```text
runs/relation_toy/quantum_steering_eval/metrics.json
runs/relation_toy/quantum_steering_eval/predictions.jsonl
runs/relation_toy/quantum_steering_eval/run_info.json
```

This is a toy quantum-inspired prototype check, not evidence for the final paper claim.

## Current Toy Spectral Filtering Sweep

After the baseline dry run succeeds, run a toy sweep over spectral filters for both classical and quantum projectors:

```bash
python experiments/sweep_relation_spectral_filters.py --model_dir runs/relation_toy --batch_size 4 --device cpu --families classical,quantum --modes hard_topk,high_pass,band_pass,soft_energy --ranks 2,4 --thresholds 0.5 --sharpnesses 8.0 --gains 0.25 --num_qubits 4 --angle_scale 1.25 --output_dir runs/relation_toy/spectral_filter_sweep
```

Expected outputs:

```text
runs/relation_toy/spectral_filter_sweep/results.jsonl
runs/relation_toy/spectral_filter_sweep/summary.json
```

Each JSONL row contains projector family, filter parameters, steered metrics, delta versus baseline, projector diagnostics, and quantum kernel diagnostics where applicable.

## Current Toy Adaptive Routing Eval

After the baseline dry run succeeds, evaluate the toy adaptive projector router:

```bash
python experiments/eval_relation_routing.py --model_dir runs/relation_toy --batch_size 4 --device cpu --gain 0.25 --temperature 0.5 --rank 2 --num_qubits 4 --angle_scale 1.25 --output_dir runs/relation_toy/relation_routing_eval
```

Expected outputs:

```text
runs/relation_toy/relation_routing_eval/metrics.json
runs/relation_toy/relation_routing_eval/run_info.json
runs/relation_toy/relation_routing_eval/routing.jsonl
runs/relation_toy/relation_routing_eval/predictions.jsonl
```

The router builds a small expert bank, computes soft expert weights from anchor representations, and applies a batch-wise dynamic projector. This is a toy routing prototype check, not a real routing benchmark.

## Real-Data Preparation Draft

The coding side has added real relation extraction format adapters. Supported formats are:

```text
project_jsonl
tacred_json
tacred_jsonl
semeval2010_task8
```

Convert a TACRED/Re-TACRED style dataset to canonical JSONL:

```bash
python experiments/prepare_relation_data.py \
  --format tacred_json \
  --dataset_name retacred \
  --train_path data/raw/retacred/train.json \
  --valid_path data/raw/retacred/dev.json \
  --test_path data/raw/retacred/test.json \
  --output_dir data/relation/retacred
```

Convert SemEval-2010 Task 8:

```bash
python experiments/prepare_relation_data.py \
  --format semeval2010_task8 \
  --dataset_name semeval2010_task8 \
  --train_path data/raw/semeval2010/TRAIN_FILE.TXT \
  --valid_path data/raw/semeval2010/TEST_FILE_FULL.TXT \
  --output_dir data/relation/semeval2010_task8
```

The converter writes canonical `train.jsonl`, `valid.jsonl`, optional `test.jsonl`, and `data_config.json`.

## Real-Data Smoke Gate

Before any full GPU benchmark, the coding side must first complete a tiny real-data smoke run:

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config data/relation/retacred/data_config.json \
  --output_dir runs/retacred_real_smoke \
  --device cpu \
  --max_train_records 256 \
  --max_valid_records 128
```

Expected outputs:

```text
runs/retacred_real_smoke/baseline/model.pt
runs/retacred_real_smoke/baseline/relation_projector.pt
runs/retacred_real_smoke/baseline/relation_quantum_projector.pt
runs/retacred_real_smoke/classical_steering_eval/metrics.json
runs/retacred_real_smoke/quantum_steering_eval/metrics.json
runs/retacred_real_smoke/spectral_filter_sweep/results.jsonl
runs/retacred_real_smoke/relation_routing_eval/metrics.json
runs/retacred_real_smoke/pipeline_summary.json
```

For now, dzy958 should not start large real-data GPU runs until the coding side provides a specific commit, dataset, and command.

## Logging

Create a log directory for each run:

```bash
RUN_DIR=runs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
python --version > "$RUN_DIR/python.txt"
python -c "import torch; print(torch.__version__)" > "$RUN_DIR/torch.txt"
python -m pytest -q 2>&1 | tee "$RUN_DIR/pytest.log"
```

## Experiment Report Template

Every report should include:

```text
Git commit:
Conda/venv name:
Python executable:
Python version:
Torch version:
GPU model:
Command:
Dataset:
Split:
Random seed:
Main metric:
Runtime:
Peak memory:
Status: success / failed
```

If failed, include:

```text
full traceback
exact command
last 100 lines of log
whether the failure is reproducible
```

## GPU Check

Before real training runs, record GPU information:

```bash
nvidia-smi
```

If `nvidia-smi` is unavailable, report that explicitly.

## Do Not

```text
do not edit source files during experiment runs
do not change committed configs without telling the coding side
do not report only screenshots; include plain-text logs
do not rerun with changed parameters without recording the exact command
```

## Re-TACRED Handoff

The concrete Re-TACRED GPU debug, low-resource, and full validation commands live in [retacred_experiment_handoff.md](retacred_experiment_handoff.md). Run them only after the coding side confirms the target commit and dataset location.

For a Chinese project-level explanation of how this differs from the previous reference-paper experiments, read [retacred_collaborator_overview_zh.md](retacred_collaborator_overview_zh.md).

If you receive `retacred_q_attention_data.tar.gz`, put it at the repository root and extract it with `tar -xzf retacred_q_attention_data.tar.gz` before running the Re-TACRED commands. The archive should create `data/relation/retacred/`.

## Current Immediate Tasks for dzy958

1. Create or activate a suitable Python environment.
2. Install the project with `python -m pip install -e ".[dev]"`.
3. Run smoke tests only when asked by the coding side.
4. Report the exact commit hash and pytest output for any smoke test.
5. Confirm GPU model using `nvidia-smi` if available.
6. Help identify and prepare candidate real datasets for relation/event/aspect extraction.
7. Do not start large GPU experiments until the coding side provides a real-data command and config.
## Re-TACRED Build Command

When the coding side confirms that licensed TACRED files are available on the machine, build Re-TACRED with:

```bash
python experiments/build_retacred_from_tacred.py \
  --tacred_dir data/raw/tacred/data/json \
  --patch_dir data/raw/Re-TACRED-source/Re-TACRED \
  --output_dir data/raw/Re-TACRED-patched-jsonl

python experiments/prepare_relation_data.py \
  --format tacred_jsonl \
  --dataset_name retacred \
  --train_path data/raw/Re-TACRED-patched-jsonl/train.jsonl \
  --valid_path data/raw/Re-TACRED-patched-jsonl/dev.jsonl \
  --test_path data/raw/Re-TACRED-patched-jsonl/test.jsonl \
  --output_dir data/relation/retacred
```

Do not commit any files under `data/`; the directory is intentionally ignored.