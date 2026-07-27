# Q-Attention

Q-Attention is a public research codebase for standalone quantum attention intervention in span-centric NLP. The current primary task is Re-TACRED relation extraction.

The project remains grounded in the reference paper's attention-steering principle, but the active method can intervene directly on attention scores instead of depending only on an offline key-space projector.

## Current Method

```text
frozen relation baseline
        -> relation-conditioned quantum attention-score kernel
        -> dual Q-RES evidence selector
        -> signed steering + positive sufficiency readout
```

The dual readout reuses one parameterized quantum state preparation:

- a signed phase-sensitive readout steers attention;
- a positive connected-projector readout measures counterfactual sufficiency;
- `context_budget` fixes the sufficiency evidence mass;
- matched local and strong classical selectors provide controlled comparisons.

The score residual also has an exact query-aligned key-update interpretation, preserving the conceptual link to key steering.

## Current Experiment

The current gate is a full Re-TACRED train/valid run with:

1. frozen baseline;
2. quantum and classical score cores;
3. quantum, local classical, and strong classical dual Q-RES selectors;
4. fixed mechanism diagnostics;
5. no access to blind test during training or selection.

Seed 13 is the full-pipeline pre-run. The formal comparison uses seeds `7, 11, 13, 17, 23`, followed by a single frozen blind-test evaluation.

No quantum-advantage claim is made from the toy or proportional-subset results alone.

## Documentation

- Documentation index: [docs/README.md](docs/README.md)
- Current method overview: [docs/current/method_overview_zh.md](docs/current/method_overview_zh.md)
- Collaborator Git workflow: [docs/current/collaborator_git_workflow_zh.md](docs/current/collaborator_git_workflow_zh.md)
- Full seed-13 run guide: [docs/current/retacred_dual_qres_full_run_zh.md](docs/current/retacred_dual_qres_full_run_zh.md)
- Formal experiment protocol: [docs/current/experiment_protocol_zh.md](docs/current/experiment_protocol_zh.md)

Historical projector, smoke-pipeline, and early plugin documents are preserved under [docs/archive/](docs/archive/README.md) and are not current run instructions.

## Code Map

```text
src/q_attention/models/relation_transformer.py       # explicit attention-score hook
src/q_attention/adapters/attention_scores.py         # score intervention adapter
src/q_attention/plugins/attention_score_kernel.py    # quantum/classical score cores
src/q_attention/plugins/attention_evidence.py        # dual Q-RES selectors
src/q_attention/plugins/attention_routing.py         # optional expert routing
experiments/train_relation_attention_score_kernel.py # core training
experiments/train_relation_counterfactual_evidence.py# selector training
scripts/run_retacred_dual_qres_full.sh                # full seed runner
scripts/export_retacred_dual_qres_report.sh           # public report exporter
tests/                                               # mechanism and regression tests
docs/                                                # current and archived documentation
```

## Installation And Check

Use an existing Python environment with PyTorch:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The active regression suite currently contains 174 tests.

## Data Policy

Re-TACRED/TACRED data, `runs/`, checkpoints, predictions, credentials, and private operational files must not be committed to this public repository. Only compact experiment reports belong under `reports/`.

## Legacy Compatibility

Legacy key steering, projector learning, spectral filtering, and expert-routing prototypes remain in the repository for ablation and historical comparison. Their old run guides are archived and do not define the active main experiment.
