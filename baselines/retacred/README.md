# Re-TACRED external references

This directory owns external baseline adapters, provenance and the shared
collaborator lifecycle. It does not contain datasets, upstream vendored code,
pretrained weights or credentials.

`cohort.json` preserves the fifteen previously discussed **reported
configurations**, not a certified global leaderboard or fifteen independent
models. A sixteenth classic entity-marker anchor is included. Four marker
variants are implemented through the exact pinned upstream `REModel` and
`RETACREDProcessor`. Normal PTR imports source forward, objective and two
optimizers (see `ptr-mechanism-contract.md` and `ptr-adaptation-diff.md`). Its
tiny CPU source/trace/resume gates passed; full-weight CUDA certification is pending.
Every unavailable or unported configuration remains
visible. `--models all` fails before launching work; it never substitutes a
generic encoder or pretends the remaining methods ran.

Sequence Routing's missing `Abstraction.py` is now recoverable from official
history without modifying the source pin. H3/Mix output512 is confirmed, but
41-vs-40 labels, paper/source budgets and metadata bootstrap remain unresolved.
Released all-label micro-F1 equals accuracy, not the project's non-null micro-F1;
the reported92.2 is not certified same-metric rank evidence. See
`sequence-routing-contract.md`. `python -m baselines.retacred.sequence_audit`
is an offline/read-only check and intentionally exits2 while admission is blocked.

PTR Reversed is a **changed-task reference**, not a canonical-task strict baseline.
The repository member explicitly distinguishes its selective inverse relations
from the conventional setting. Its reported91.9 remains visible but is rank-ineligible.
See `ptr-reversed-contract.md`; `python -m baselines.retacred.ptr_reversed_audit`
is an offline audit, expected exit2, no model jobs or normal-PTR substitution.

## Mechanism vs experiment protocol

Per-model manifests, `mechanism-contract.md` and `adaptation-diff.md` freeze
the source facts and evidence ceiling. Full-weight formal certification remains
`not_verifiable` until real-weight CUDA/AMP and common-protocol gates are
recorded; a tiny fixture or imported class is not a formal certificate.

- The active source is imported, not rewritten: marker tokenization,
  subject/object marker pooling, concatenation, upstream MLP and cross-entropy.
- A source-fidelity gate compares upstream with the adapter at fixed input:
  forward, loss, gradients, AdamW update and tracing transparency.
- The **common adapted training protocol** is separate from an exact replay of
  published scores. It uses fixed logical batches (64 by default), weighted
  physical micro-batches, final-epoch selection and no intermediate test access.
  It does not retain upstream's first-batch/off-by-one accumulation update,
  dropped tail or periodic test evaluation. These are declared adaptations.
- Marker configurations: five epochs, LR 3e-5, Adam epsilon 1e-6, weight decay 0, bias correction,
  gradient clip 1 and 10% linear warmup are frozen. CUDA retains upstream
  autocast and GradScaler. No silent PyTorch-AdamW fallback.
- PTR retains source FP32, five verbalizer CE plus relation CE, dual AdamW
  LR3e-5/1e-5, encoder decay.01 with bias/LayerNorm exclusions, extra decay0,
  epsilon1e-6 and warmup500. Its extra-token flags stay zero. This differs from
  marker optimization; a common budget is not an identical source recipe.
- Physical micro-batch size and checkpointing are immutable resume controls:
  dropout prevents a claim of bit-exact equivalence after changing the partition.
  GPU topology may change because each `(model,seed)` is an isolated task. OOM
  stops that task and preserves its last durable checkpoint; this release does
  **not** silently change physical batches or kill unrelated processes.

## Observability and privacy

One predeclared non-null train sample and one non-null test sample are captured
from their actual computation. Leaf module input/output tensors (embeddings,
all transformer layers, Q/K/V projections, attention, classifier), activation
gradients and aggregated classifier parameter gradients are detached to CPU.
No second forward reconstructs a claimed training event. JSON records raw
sentence, tokens, entity spans, marker positions, target/prediction, module
I/O, tensor shapes and full tensor artifact identity. Loss scaling and
parameter-gradient aggregation are explicitly distinguished.

Actual numeric `.pt` files and private trace JSON travel over authenticated
18084 exchange; public Git contains only compact hash-verified references.
Files above the service's 1 GiB limit are sharded. All shards are validated
before reassembly. The service does not support HTTP byte-range resume:
completed shards are reused, partial shards retry from the beginning.

## Layout

- `assets.py`: official pinned RoBERTa weights, sharding, authenticated downloads.
- `common.py`: source pinning, identity, complete benchmark/all-label metrics.
- `improved.py`: upstream wrapper and real-batch tensor hooks.
- `fidelity.py`: fixed-input runtime equivalence gate.
- `ptr_resources.py`: recovered official historical metadata.
- `ptr.py`, `ptr_fidelity.py`: normal PTR source adapter and runtime gate.
- `ptr_reversed_audit.py`: pinned-source/changed-task audit, not a reversed adapter.
- `sequence_resources.py`: isolated hash-locked historical file recovery, not an adapter.
- `sequence_audit.py`: static source/recipe/metric audit; no import or training side effects.
- `updates.py`: mandatory named dual optimizer/scheduler recovery.
- `adapters.py`: explicit allowlist of source adapters, no silent substitutions.
- `train.py`: logical-batch trainer and post-update recovery.
- `orchestrate.py`: one heavy worker per GPU, safe pause, dynamic queue.
- `report.py`: mandatory tensor upload, atomic whitelist export, commit/push.

Formal full-data training is collaborator-run only. See
`docs/current/retacred_reference_suite_zh.md` for the one-command handoff.
