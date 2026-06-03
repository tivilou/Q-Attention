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
finish NLP encoder adapter
add relation extraction data schema
prepare first baseline command
```

dzy958:

```text
confirm available GPU and conda environment
confirm preferred first relation extraction dataset
run phase-1 demo and tests from the latest main branch
report environment issues
```

## Runner Guide

Detailed run commands and reporting templates live in [experiment_runner_guide.md](experiment_runner_guide.md).
