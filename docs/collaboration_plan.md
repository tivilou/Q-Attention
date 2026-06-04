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
finish the toy-data classical, quantum, spectral-filtering, and adaptive-routing loops
verify baseline -> projector bank -> spectral sweep -> routed steering eval commands
add tests and reproducibility docs
prepare the next real-data task adapters before formal GPU handoff
```

dzy958:

```text
confirm available GPU and Python environment
keep environment details ready for future runs
help identify candidate real datasets for relation/event/aspect extraction
run only smoke or toy commands when explicitly asked
wait for real-data commands before large GPU experiments
```

## Current Code Handoff Status

```text
prototype status: toy-data classical, quantum, spectral-filter, and adaptive-routing verification only
formal experiment handoff: not yet
next gate: real dataset loader + config + command must pass a small real-data smoke run
```

## Runner Guide

Detailed run commands and reporting templates live in [experiment_runner_guide.md](experiment_runner_guide.md).
