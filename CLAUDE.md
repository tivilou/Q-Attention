# CLAUDE.md

## Project Goal

This repository is for an original quantum machine learning attention-steering project for span-centric NLP information extraction. The method should inherit the principle of spectral key steering while keeping all public source code clean-slate.

## Core Principle

The project should preserve this mechanism:

```text
learn projector P from key-space contrastive structure
apply k' = k + gPk inside attention layers
leave model weights unchanged
```

The project should not become a generic quantum-attention paper detached from this mechanism.

## Working Rules

- Do not copy external project source code into this repository.
- Write the implementation from scratch.
- Keep the public repository focused on original code, design notes, and application experiments.
- Use real NLP task metrics, not only attention visualization.
- Keep first prototypes lightweight and torch-only unless a quantum framework is explicitly required.

## Research Themes

1. Quantum-kernel projector learning.
2. QSVT-inspired spectral projector filtering.
3. Quantum adaptive expert routing for key steering.
4. Practical deployment in span-centric NLP information extraction.

## Current Mainline

- The primary method is a standalone supervised quantum projector.
- It must generate `P_q` without requiring a classical projector.
- Use parameterized entangling quantum features and relation-label kernel alignment.
- Keep classical-plus-quantum residual projectors as ablations only.
- Preserve `k' = k + gP_qk` as the injection mechanism.

## Preferred Application Track

Focus on span-centric information extraction:

```text
Relation Extraction
Event Argument Extraction
Aspect-Based Sentiment Analysis
Biomedical Relation Extraction
```

These tasks are appropriate because they require attention models to focus on anchor and evidence spans while making structured predictions.


## Collaboration Workflow

The coding side prepares implementation, tests, scripts, and reproducible commands. The collaborator `dzy958` runs experiments, records logs, collects metrics, and reports failures with full tracebacks. Keep experiment commands simple enough to run without editing source code.

## First Implementation Target

Start with relation extraction as the first task because it has a clear anchor structure:

```text
text + entity pair -> relation label
```

Then generalize the same mechanism to event arguments, aspect sentiment, and biomedical relations.

Initial modules should include:

```text
build_projector(...)
apply_key_steering(...)
quantum_kernel_projector(...)
span_anchor_adapter(...)
```
