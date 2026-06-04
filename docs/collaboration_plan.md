# Collaboration Plan

## Roles

```text
tivilou / coding agent: implementation, tests, scripts, documentation, reproducibility
dzy958: experiment execution, dataset preparation, logs, result collection, error feedback
```

## Coding Responsibilities

The coding side is responsible for:

```text
core method implementation
unit tests
minimal demos
training and evaluation scripts
configuration files
result directory conventions
bug fixes based on experiment logs
real-data adapters and smoke-run gates before GPU handoff
```

## Experiment Responsibilities for dzy958

The experiment side is responsible for:

```text
setting up conda environments
preparing datasets
running provided commands without modifying core code
recording exact command lines
saving logs and metrics
reporting failures with full traceback
summarizing result tables
```

## Expected Experiment Report Format

Each experiment report should include:

```text
git commit hash
conda environment name
GPU model
command line
random seed
dataset split
metric output
runtime
peak memory if available
error traceback if failed
```

## Current Immediate Task Split

Coding side:

```text
finish real relation dataset adapters for TACRED/Re-TACRED style data and SemEval-2010 Task 8
verify converter -> small real-data smoke pipeline -> baseline/projector/sweep/routing outputs
add tests and reproducibility docs
freeze a specific handoff command only after the real-data smoke gate passes
```

dzy958:

```text
confirm available GPU and Python environment
keep environment details ready for future runs
help identify and prepare candidate real datasets for relation/event/aspect extraction
run only smoke or toy commands when explicitly asked
wait for a frozen real-data command before large GPU experiments
```

## Current Code Handoff Status

```text
prototype status: toy-data classical, quantum, spectral-filter, and adaptive-routing verification complete
real-data status: converter/config/smoke-pipeline scaffold in progress
formal experiment handoff: not yet
next gate: selected real dataset must pass a small smoke run with pipeline_summary.json and all expected result files
```

## Runner Guide

Detailed run commands and reporting templates live in [experiment_runner_guide.md](experiment_runner_guide.md).

## Real-Data Smoke Plan

The real-data gate and converter commands live in [real_data_smoke_plan.md](real_data_smoke_plan.md).