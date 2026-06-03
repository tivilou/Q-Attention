# Experiment Runner Guide

This guide is for `dzy958`, who is responsible for running experiments, collecting logs, and reporting results.

## Role

The experiment runner should not modify core source code during a run. The expected workflow is:

```text
pull latest main
check environment
run the provided command
save logs and metrics
report success or failure with exact details
```

## Repository

```text
https://github.com/tivilou/Q-Attention
```

Server working copy currently used by the coding side:

```text
/home/Q-Attention/Q-Attention-public
```

## Conda Environment

The current server validation uses:

```text
conda env: py310
python: 3.10
```

Check environment:

```bash
/usr/local/miniconda3/bin/conda run -n py310 python --version
/usr/local/miniconda3/bin/conda run -n py310 python -c "import torch; print(torch.__version__)"
```

## Pull Latest Code

```bash
cd /home/Q-Attention/Q-Attention-public
git pull
```

Record the commit before running anything:

```bash
git rev-parse HEAD
```

## Current Smoke Tests

Run these after every pull:

```bash
cd /home/Q-Attention/Q-Attention-public
/usr/local/miniconda3/bin/conda run -n py310 env PYTHONPATH=src python examples/minimal_key_steering.py
/usr/local/miniconda3/bin/conda run -n py310 env PYTHONPATH=src python examples/encoder_adapter_demo.py
/usr/local/miniconda3/bin/conda run -n py310 env PYTHONPATH=src python -m pytest -q
```

Expected result:

```text
minimal_key_steering demo passes
encoder_adapter demo passes
pytest reports all tests passed
```

Current expected pytest count:

```text
9 passed
```

## Logging

Create a log directory for each run:

```bash
mkdir -p runs/$(date +%Y%m%d_%H%M%S)
```

Recommended pattern:

```bash
RUN_DIR=runs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
/usr/local/miniconda3/bin/conda run -n py310 env PYTHONPATH=src python -m pytest -q 2>&1 | tee "$RUN_DIR/pytest.log"
```

## Experiment Report Template

Every report should include:

```text
Git commit:
Conda env:
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

## Current Immediate Tasks for dzy958

1. Confirm the conda environment can run the three smoke-test commands.
2. Report the exact commit hash and pytest output.
3. Confirm GPU model using `nvidia-smi`.
4. Confirm which relation extraction dataset should be prepared first.
5. Send any failure logs back to the coding side for fixes.
