# LUKE training/evaluator binding audit

Status: preparation only, not_verifiable. No formal worker, cohort admission or
paper score follows from this audit. The official source is pinned at
251654548a15b9a8ddc708a53b715bdf3a112102; existing foundation assets/adapter are
unchanged. Source files stay private in .external, not vendored into Git.

## Newly executable scorer contract

Official `examples/relation_classification/metrics/multiway_f1.py` has SHA256
9992976c88a3b9bca8b549486320fcc44cfe8950ea14acc30bbc10171585fc72.
`luke_recipe_audit.py` verifies these bytes before executing only the locked
default-label assignment, normalization function and scorer class as AST.
Python list/Counter/equality/mean carriers replace framework-only operations.
This establishes scorer algebra and reset behavior, not AllenNLP/Torch runtime.

- Source micro-F1 excludes `no_relation` from positive counts but keeps every
  example, including false positives on null relations. This matches the common
  benchmark non-null micro-F1 count law on the declared scalar fixtures.
- Source macro-F1 averages the union of observed gold/predicted non-null labels,
  not all40 labels and not `no_relation`. This is different from the project's
  all40-label macro-F1 with zero division0. Do not relabel one as the other.
- All-null source macro is undefined/NaN; the audit reports this explicitly,
  never silently changes it to zero or writes NaN into a JSON metric.
- Default KBP37 normalization removes `(e1,e2)`/`(e2,e1)` suffixes before count
  grouping, but correctness still uses original label-ID equality. Canonical
  relation fixtures remain unchanged. No new direction-merging rule is added.
- Accumulation and reset are exercised directly on the locked scorer methods.

For a three-example fixture gold=[null,A,B], prediction=[null,A,null], source
micro-F1=2/3 and source observed-non-null macro-F1=1/2. Using three named labels
plus37 explicitly unused synthetic labels, common all40 macro-F1=1/24. These are
synthetic evaluator counterexamples, not measured model results.

## Hash-bound configuration and framework recipe

`luke_recipe.lock.json` binds18 selected source files. All eight LUKE files
match official Git blobs at the immutable revision. The ten AllenNLP members
come from the official2.9.2 sdist, whose SHA256 is exactly the value in that
LUKE revision's poetry.lock:
004d6a4ad64891779d90f9c407b49141a5214aef0c6cc0794dba776f9927e687.
Private source bytes stay in .external; no upstream package is installed.

The Jsonnet import closure is the entity-aware entry, base.libsonnet and the
entity-aware model fragment; the two imported fragments have no further imports.
The audit extracts locked literal objects and environment names, not a Jsonnet
evaluation. TRAIN_DATA_PATH, VALIDATION_DATA_PATH, DATASET and
TRANSFORMERS_MODEL_NAME are required. README supplies DATASET=tacred and the
LUKE-large checkpoint; switching to qualified canonical Re-TACRED, changing
label order, or using project seed13 is a declared task/common-protocol adaptation,
not the author's original TACRED seed0 result.

Bound source configuration:5epochs, physical batch4, shuffle=true,
accumulation8, AdamW1e-5, decay.01 except bias/LayerNorm.weight, warmup.06,
patience3, dev +micro_fscore, all three RNG seeds0. poetry.lock binds
AllenNLP2.9.2/Torch1.10.2/Transformers4.13.0; the existing project environment
has different Torch/Transformers versions and is not silently replaced.

The registered adamw source wraps torch.optim.AdamW, not the separately
registered huggingface_adamw. Bound defaults: betas(.9,.999), eps1e-8,
amsgrad=false; trainer grad_norm=false and grad_clipping=None. Base config uses
CPU and AMP defaultsfalse; the author's README explicitly overrides CUDA0 and
use_amp=true. Source grad_scaling defaultstrue when AMP is used. CUDA overflow
and scaler/optimizer behavior still require actual framework/runtime tests.

The actual trainer passes ceil(number_of_physical_batches/8) as optimizer
steps per epoch, including an incomplete group. The custom scheduler computes
int(5*steps_per_epoch*.06) warmup steps, uses polynomial power1 and endLR0.
Its initialization sets step0; the active trainer calls step_batch(group_index+1)
BEFORE optimizer.step. Do not infer the ordering from the generic scheduler
docstring, which describes a different after-update order.

The source averages each physical batch's mean loss by the actual number of
batches in the group, including its tail, not by a fixed8 and not by example
count. A fixture with batch sizes[4,4,2] and mean losses[1,2,3] gives source
loss2.0 versus example-weighted1.8. Batch partition changes therefore require
an explicit adaptation gate; nominal effective batch32 is not enough.

Source MetricTracker uses strict improvement; ties preserve the earlier best
epoch and count towards three consecutive non-improving validation epochs.
Locked scheduler/tracker classes are executed with scalar optimizer/registry
carriers. Their counting, floating LR values and scalar state roundtrips are
verified. These are not neural optimizer updates, model checkpoints, AMP,
AllenNLP loader/indexer execution or reproduction of a published score.

## AllenNLP interface boundary

The pinned reader returns word_ids/text-field tokens, subject/object inclusive
SpanFields and role-ordered [MASK]/[MASK2] entity IDs. LabelField is separate.
The entity extractor and HF conversion use subject then object entity readout,
concatenation, dropout and linear classification; CLS replacement is forbidden.
The adapter uses real HF BPE, explicit inclusive positions with -1 padding and
new canonical40 head. It does not import an actual AllenNLP tokenizer/indexer or
loader. Scalar field carriers and real HF tokenizer fixtures are preparation,
not a certificate of that framework boundary.

Classifier initialization law, RNG ordering, label-ID order, max-length rejection
instead of cropping, batch collation, physical partition and checkpoint selection
remain explicit adaptations. Source-versus-adapter numerical evidence and
common-protocol approval are required before formal admission. No CoRE/CFIE,
resampling or focal mechanism is inferred from a base LUKE recipe.

## Audit entry

`python -m baselines.retacred.luke_recipe_audit --source-cache PINNED_SOURCE_CACHE`
runs only offline scalar scorer checks and prints evidence. Exit2 means metric
algebra verified but separate recipe/framework/full-weight admission pending;
exit1 means invalid source or failed checks. It performs no model allocation,
network request, persistent file write, training, upload or Git operation.

`bash scripts/check_retacred_luke_recipe.sh --describe` displays scope without
Python/source/model access. The default command uses the existing private
source cache; `--source-cache PATH` selects an explicit hash-locked cache.
PYTHON_BIN overrides the portable python/python3 selection. It prints JSON and
returns2 for source_recipe_bound_neural_runtime_pending, or1 for invalid sources
or failed checks. It never downloads, installs, runs a formal job or changes Git.
The metric and recipe audits are separate: one audit passing does not certify
the other, and neither opens the four LUKE variant formal-admission gates.
