# Fixed-feature experiment extension boundary

## One entry, existing contracts

```bash
python experiments/q_rmea_train.py --experiment protected-context unit --output /path/to/unit
python experiments/q_rmea_train.py --experiment entity-scattering --help
```

Actions and their existing arguments are forwarded in order to
`FrozenFeatureTraining.main(argv)`. Existing study-specific CLIs remain valid.
The example is a unit action, not formal admission or permission to resume L5.
The runner still enforces each study's exact review, input and run contract.

## Responsibilities

- `rmea_frozen_training.py`: established training/worker/resume lifecycle.
- `rmea_task_runtime.py`: shared microbatch math, guarded worker assignment,
  minimal preparation and immutable report export. No scientific condition switch.
- `rmea_experiments.py`: allowlisted protocol/runtime pairs, lazy imports.
- `rmea_*_runtime.py`: model factory plus scientific capture/validation/summary.
- Model modules: scientific transformations; no scheduler, COS or Git clients.

For an ordinary hypothesis sharing the task protocol, add its model plugin and
adapter/config registration. Reuse unchanged callbacks and common task functions;
do not copy the whole runtime. If evaluation, state or training assumptions differ,
make that difference explicit instead of treating it as a harmless plugin option.

## Verification reuse

Changed plugin: interface/finite output/gradient/one real update where applicable.
Changed worker/checkpoint/backend/export: the affected capability test, not an
unconditional serial + grouped resume + COS restoration matrix. Preserve complete
source/config provenance separately from relevant capability-test dependencies.
The refactor was checked using stdlib-only wiring tests and exact AST comparison
of unchanged scientific functions and moved microbatch mathematics. It is not a
new full-training or cross-provider recovery certificate.

## Frozen acceptance remains frozen

The protected-context 1f6e2e00 canary's primary checks passed but its extra RNG
acceptance failed. That historical result is unchanged. This refactor does not
forge its missing acceptance, install a formal recipe, bypass whole-source gates,
or migrate the existing training-review/canary schema. New source will fail those
old exact-source gates until an explicit scoped admission/migration is completed.
Do that once at the changed core boundary, then reuse its applicable capability
coverage; do not repeat it for every idea. The later scoped v2 migration preserves that failed historical record and binds current provenance, unchanged exercised capabilities, focused RNG repair and pinned inputs. See `docs/evidence/rmea/20261122-scoped-training-admission/`. Subsequent ordinary plugins reuse unaffected coverage; the current owner authorizes full runs through the shared scheduler.
