# Re-TACRED external references

This directory owns external baseline adapters, provenance and the shared
collaborator lifecycle. It does not contain datasets, upstream vendored code,
pretrained weights or credentials.

## Bounded runtime handoff

`bash scripts/check_retacred_reference_runtime.sh --upload` serially prepares
and checks the LUKE foundation and MetaEntail MNLI foundation. It reuses the
locked asset clients, prepares only the pinned LUKE source files, and validates
child exit/status, train/test numeric artifact identities and recovery evidence.
Resource or dependency gaps exit2; failed checks/delivery exit1; diagnostic-only
success exits0. `--describe` is read-only; `--offline` disables acquisition;
`--upload-only RUN_DIRECTORY` retries evidence delivery without model execution.
Private artifacts are hash-declared and large tensors/checkpoints are sharded.
This entry never launches formal training, changes Git or admits any of the
blocked methods. See `docs/current/retacred_reference_runtime_zh.md`.

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

Curriculum Learning's paper model includes **dependency GAT + bilinear
+classifier**, not marker-only training with a sorting callback. Cross reviewers
+train on their own subsets. Its reported91.4 is +0.2 percentage points over
+its own91.2 baseline. Source identity, equations, pacing and scorer are unresolved;
+see `curriculum-contract.md`. `python -m baselines.retacred.curriculum_audit`
+is offline/read-only, expected exit2, no model launch or substitute.
+
+LUKE/CFIE original author source was recovered on the **RE** branch, not its
+default NER branch. The released CoRE example is a gold-conditioned TACRED
+challenge **macro** analysis, not the paper's full Re-TACRED **micro** result.
+The exact LUKE ports/training recipes remain blocked; see
+`luke-variants-contract.md`. `python -m baselines.retacred.luke_audit` is an
+offline audit (exit2), executes only scalar scorer/selection AST fixtures,
+and never loads a model, pickle or dataset. Default five models stay unchanged.
+
+## Mechanism vs experiment protocol

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
- `luke_audit.py`: source/score/port audit, not a LUKE training adapter.
- `curriculum_audit.py`: paper/source readiness audit, not a curriculum adapter.
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

## LUKE foundation preparation (not formal admission)

`luke.py` executes the pinned author's parser and HF classifier; `luke_fidelity.py`
checks source/head interface fixtures without loading Torch or weights.
`luke_foundation.lock.json` pins official source and checkpoint asset identities.
See `luke-foundation-contract.md` and `luke-foundation-adaptation.md`.
Real backbone/tokenizer/update/resume/trace/CUDA gates remain pending. Four LUKE
variants stay blocked; this support module adds no new cohort model or default job.

## LUKE pretrained asset mirror

`luke_assets.py` and `luke_assets.lock.json` cover all eight pinned luke-large
assets. `scripts/fetch_retacred_luke_assets.sh` performs authenticated HTTPS
fetch, shard checksum verification, atomic reconstruction and original asset
identity validation. All twelve parts are <=512MiB. Exchange URL/token are
environment-only, not in the asset lock. No Torch/model loading occurs.
Cached complete files are rechecked and need no credentials/network. Corrupt
existing files are preserved and rejected, never silently replaced.

This asset proof does not certify the LUKE forward, optimizer or formal recipe.
Historical foundation lock preparation flags describe the earlier snapshot;
current materialized asset evidence is the separate luke_assets.lock.json.
The default five configurations and blocked four LUKE variants are unchanged.

## Offline real LUKE tokenizer gate

`luke_tokenizer_fidelity.py` loads the seven hash-verified tokenizer assets
locally, never the weight file. In a dedicated process it disables model
backends and blocks their imports. It checks actual HF BPE IDs against the
pinned author's `text_to_instance` span method on list-only field carriers.
Fixtures cover reversed/adjacent roles, punctuation, Unicode, subword expansion,
long mentions, mixed-batch padding, label isolation and the512-token boundary.
The optional evidence file contains complete fixture IDs/tokens/positions.

This is a tokenizer/interface proof, NOT actual AllenNLP wrapper/indexer or
complete model runtime equivalence. No cohort admission or manifest promotion
follows from this gate. The source preparation snapshot remains historical.


## Owner-approved practical reproduction scope

Missing author implementation details may use reasonable frozen, explicit
choices. Such methods are separately labeled paper-interpreted references,
not exact author implementations or strict-fidelity certificates. Core
mechanisms, heldout isolation and actual runtime evidence remain mandatory.

PTR Reversed is now excluded_noncomparable/run_eligible=false: the historical
row is retained, but --models all omits it and explicit requests are rejected
before any job. The execution target currently has15 configurations including
the classic anchor; five implemented defaults remain unchanged. More exclusions
require demonstrated task incompatibility, not simply missing source details.

CTL-DRP now has an actual injected-encoder scalar Eq2 neural core, real BPE and
four paper marker tokens, synthetic train/test numeric captures and CPU FP32
post-update recovery. `check_retacred_ctl_drp_neural.sh --describe` is backend-
free; execution requires explicit private output/cache/native-source paths.
See ctl_drp_runtime_choices.json for actual tiny scope and explicit full-recipe
proposals. Large pretrained loading and full task integration are still pending.
This diagnostic neither downloads assets nor runs benchmark/full-data training.
