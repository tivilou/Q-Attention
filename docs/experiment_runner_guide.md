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
python -m pytest -q
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

## Current Immediate Tasks for dzy958

1. Create or activate a suitable Python environment.
2. Install the project with `python -m pip install -e ".[dev]"`.
3. Run the three smoke-test commands.
4. Report the exact commit hash and pytest output.
5. Confirm GPU model using `nvidia-smi` if available.
6. Confirm which relation extraction dataset should be prepared first.
7. Send any failure logs back to the coding side for fixes.
