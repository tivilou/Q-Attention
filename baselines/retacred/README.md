# Re-TACRED external references

This directory owns external baseline adapters, provenance and the shared
collaborator lifecycle. It does not contain datasets, upstream vendored code,
pretrained weights or credentials.

`cohort.json` preserves the fifteen previously discussed **reported
configurations**, not a certified global leaderboard or fifteen independent
models. A sixteenth classic entity-marker anchor is included. Four marker
variants are implemented through the exact pinned upstream `REModel` and
`RETACREDProcessor`. Every unavailable or unported configuration remains
visible. `--models all` fails before launching work; it never substitutes a
generic encoder or pretends the remaining methods ran.

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
- Five epochs, LR 3e-5, Adam epsilon 1e-6, weight decay 0, bias correction,
  gradient clip 1 and 10% linear warmup are frozen. CUDA retains upstream
  autocast and GradScaler. No silent PyTorch-AdamW fallback.
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
- `ptr_resources.py`: recovered official historical metadata; PTR adapter pending.
- `train.py`: logical-batch trainer and post-update recovery.
- `orchestrate.py`: one heavy worker per GPU, safe pause, dynamic queue.
- `report.py`: mandatory tensor upload, atomic whitelist export, commit/push.

Formal full-data training is collaborator-run only. See
`docs/current/retacred_reference_suite_zh.md` for the one-command handoff.
