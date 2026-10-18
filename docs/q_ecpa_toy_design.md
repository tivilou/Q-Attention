# Q-ECPA toy/preflight v1

Scope: bounded CPU synthetic relation task, not a formal Re-TACRED experiment.
Owner authorized implementation after the 2026-10-17 evidence-grounded proposal.
Source baseline: origin/main 4bef635cfee7f10e196543067936b09b2bfa3479.

## Frozen mechanism

Use normalized head queries z (not labels) as current token features. Pool z
over the supplied ordered subject/object masks, concatenate them, apply a
shared Linear(2D,D²), and antisymmetrize its output. Phase is
pi*tanh(z_i^T A(r) z_j), explicitly antisymmetrized for numerical precision.
Adjacent/skip-one valid tokens form a fixed symmetric graph, including triangles.
Degree-normalized complex edges form a Hermitian H[destination,source].
U=exp(-0.7iH); retain all three propagation steps. Coherent prior is
abs(U³)² transposed to query/key order. The same-H dephased prior is
(abs(U)²)³ transposed. Valid-token log prior is row-centered, clipped to ±4,
and multiplied by fixed alpha=.1 before adding to baseline logits. Fixed
nonzero alpha avoids a dead initialization; key/query padding receives no action.
Singleton/isolated graph nodes are valid identity-propagation states.

Explicit allowlist: disabled, quantum, dephased, classical, random_phase.
The classical comparator uses the SAME trainable entity-conditioned phase
features as real edge logits, self edges, and three real softmax transitions.
This matches Q/D/classical parameter budgets, conditioning, depth and residual
interface; it does not assert equal FLOPs or identical function classes.
Random_phase freezes a seeded random antisymmetric operator and removes entity
conditioning; token features remain learned. Its trainable budget is explicitly
lower. It is not an entanglement/parity proof or a parameter-matched control.

Old Q-WAP is closed. That module used a frozen score-derived real symmetric
graph, bounded context residual and a different invalidated partial-evidence
benchmark. This implementation does not reopen those variants. The differences
here are learned ordered-entity phases, loop flux, pre-softmax intervention and
a fresh label-free benchmark. No broad novelty claim is made: quantum-walk
attention already exists. Existing original QK is included in an executable
distinctness/control check, not treated as a matched-capacity new path model.
An original-QK held-out training comparison remains a separate follow-up before
formal experiment freeze; this first screen focuses on the path mechanism.

## Toy and decision rules

Public synthetic data: two relation phrases per sentence, observed entity spans
choose which phrase is relevant; 192/48/48 train/valid/test records. The graph
never sees gold. Ordered entity triples and queried full sentences are disjoint
across splits; vocabulary is shared. Seed13, one 16-dimensional encoder layer,
two heads, FF32, dropout0, Adam .003, 100 optimizer updates, batch16 except the
first single-case update shared by every condition. No hyperparameter search.
Report test Macro-F1 at the best validation Macro-F1 checkpoint plus accuracy,
micro precision/recall/F1 (all equal here for single-label all-class scoring),
loss, per-class metrics and confusion. Test is evaluated only after training.

Mechanism gates: exact disabled/baseline equivalence; Hermiticity/unitarity;
normalized Q/D priors; m=1 Q=D; node-gauge invariance; nonzero cycle flux and
finite-difference-checked nonzero intervention gradients; padding invariance;
entity swap changes the operator; same-H Q/D control; original QK distinctness.
Toy improvement is exploratory, not L1. An ordinary toy null is inconclusive;
do not close the mechanism or tune using the held-out test. Quantum-versus-
classical superiority and formal utility are separate claims.

## Full selected-case evidence

Freeze first ID of each split before training. Capture initial/best/final for
each split in replay lane, and train-0000 at the real first optimizer update in
training lane. Every item in the code-owned inventory has complete detached
NPY values, shape/dtype/axis semantics, hash, byte count and coordinate reader.
Inventory covers IDs/masks, token/position/summed embeddings, Q/K/V, graph
edges/degrees, entity operator and phases, complex H/U, EVERY coherent/dephased/
classical step, selected prior, score residual, original/adjusted scores,
attention, context, projected attention, residual/FFN/output hidden states,
pooled entity/context features, classifier hidden state, logits/probabilities,
loss, full classifier and intervention gradients. Disabled graph states are
not applicable; its shared inference tensors still have full values.

Stages explicitly link producer outputs to subsequent inputs. Gold is a source
input only to post-prediction loss/backward, never generation. All exposed roles
and numeric representations have distinct explanations; audit hashes are not
reader-facing fields. Public synthetic evidence may be projected; real private
records require a different privacy adapter. Reader resolves arbitrary prefix
axes (head/step etc), then bounded matrix/vector/coordinate values in real,
imaginary or magnitude mode. This turn provides producer and reader, not a new
research-portal UI. Formal readiness must additionally verify the actual portal
browser's tensor projection rather than declaring it ready from metadata.

Content gate independently decodes every artifact, checks complete counts,
hashes and first/last coordinates in all complex modes, and verifies numerical
embedding and attention-to-context lineage. Shared sample-trace.v2 structural
validation is an additional gate. Full initial/best/final × split coverage and
the real training lane are mandatory. A missing item prevents RUN_COMPLETE.

## Lifecycle and limitations

Portable shell entry resolves its own checkout root. CPU-only bounded runner
rejects >500 updates; one layer, zero dropout and single-process adapter are
explicit constraints. No GPU, DDP, OOM fallback or formal-data support claimed.
Latest (full optimizer/RNG/next-update/source/config/dataset identity) and best
(model/update/metric) are the only checkpoint filenames. Both use atomic rename.
Stateless per-update sampling preserves the next batch after restart; latest
is written at each optimizer boundary. Resume accepts identical source/config
only; no historical checkpoint migration. Completed traces/metrics can be
revalidated without training; export failure after training resumes from the
last durable update. Interruption and corrupted-value tests exercise behavior.
Never run the same output directory concurrently. Formal checkpoint cadence,
large-sequence matrix-exponential cost and original QK training control remain
follow-up work; no collaborator handoff is implied by this toy runner.

Run on the authorized development server:

```bash
PYTHON_BIN=/usr/local/miniconda3/envs/py310/bin/python \
  bash scripts/run_ecpa_toy.sh --output runs/ecpa_toy/seed13
# Restart exactly the same experiment:
PYTHON_BIN=/usr/local/miniconda3/envs/py310/bin/python \
  bash scripts/run_ecpa_toy.sh --output runs/ecpa_toy/seed13 --resume
```
