# Q-Attention

Q-Attention is a public research codebase for standalone quantum attention intervention in span-centric NLP. The current primary task is Re-TACRED relation extraction.

The project remains grounded in the reference paper's attention-steering principle, but the active method can intervene directly on attention scores instead of depending only on an offline key-space projector.

## Current Method

```text
frozen relation baseline
        -> relation-conditioned quantum attention-score kernel
        -> Q-NESS evidence selector
        -> independent necessity + sufficiency residual
```

Q-NESS reuses one relation-token quantum state preparation but uses independent, non-complementary observable banks:

- necessity measures connected `Z_relation Z_token` evidence;
- sufficiency measures `X_relation Z_token` evidence;
- the two observables are noncommuting and do not use `drop = 1 - keep`;
- commuting, separable, phase-scrambled, dephased, and matched classical controls are explicit.

The score residual also has an exact query-aligned key-update interpretation, preserving the conceptual link to key steering.

## Current Experiment

The current gate is a full Re-TACRED train/valid run with:

1. frozen baseline;
2. quantum and classical score cores;
3. Q-NESS and its mechanism controls;
4. fixed mechanism diagnostics;
5. no access to blind test during training or selection.

Seed 13 is the full-pipeline pre-run. The formal comparison uses seeds `7, 11, 13, 17, 23`, followed by a single frozen blind-test evaluation.

The five-seed Q-NESS toy mechanism gate has passed. It does not establish task-level improvement or quantum advantage; proportional and full Re-TACRED validation remain separate stages.

## Documentation

- Documentation index: [docs/README.md](docs/README.md)
- Current method overview: [docs/current/method_overview_zh.md](docs/current/method_overview_zh.md)
- Q-NESS proportional gate: [docs/current/retacred_qness_proportional_gate_zh.md](docs/current/retacred_qness_proportional_gate_zh.md)
- Collaborator Git workflow: [docs/current/collaborator_git_workflow_zh.md](docs/current/collaborator_git_workflow_zh.md)
- Full seed-13 run guide: [docs/current/retacred_dual_qres_full_run_zh.md](docs/current/retacred_dual_qres_full_run_zh.md)
- Formal experiment protocol: [docs/current/experiment_protocol_zh.md](docs/current/experiment_protocol_zh.md)
- Q-NESS toy gate: [docs/current/qness_toy_gate_zh.md](docs/current/qness_toy_gate_zh.md)

Historical projector, smoke-pipeline, and early plugin documents are preserved under [docs/archive/](docs/archive/README.md) and are not current run instructions.

## Code Map

```text
src/q_attention/models/relation_transformer.py       # explicit attention-score hook
src/q_attention/adapters/attention_scores.py         # score intervention adapter
src/q_attention/plugins/attention_score_kernel.py    # quantum/classical score cores
src/q_attention/plugins/attention_evidence.py        # Q-NESS and historical Q-RES selectors
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

The active regression suite currently contains 194 tests.

## Data Policy

Re-TACRED/TACRED data, `runs/`, checkpoints, predictions, credentials, and private operational files must not be committed to this public repository. Only compact experiment reports belong under `reports/`.

## Legacy Compatibility

Legacy key steering, projector learning, spectral filtering, and expert-routing prototypes remain in the repository for ablation and historical comparison. Their old run guides are archived and do not define the active main experiment.
