# Q-ECPA bounded toy/preflight — 2026-10-17

Result: implementation and scoped evidence gates passed. Toy task utility is
positive versus disabled; quantum-specific advantage is **not** established.
This is not formal Re-TACRED evidence and does not receive L1.

| Condition | Best validation update | Test Macro-F1 | Test accuracy | Test loss |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 100 | 0.978494624 | 47/48 | 0.055914048 |
| Q-ECPA quantum | 90 | 1.000000000 | 48/48 | 0.033634309 |
| Same-H dephased | 80 | 0.978494624 | 47/48 | 0.081993274 |
| Matched classical | 100 | 1.000000000 | 48/48 | 0.013836867 |
| Fixed random operator | 90 | 1.000000000 | 48/48 | 0.028002454 |

The test set contains only48 records. The Q-baseline accuracy difference is one
prediction; a near-ceiling score is a weak basis for attributing the effect.
The classical and random controls match quantum, so the learned ordered-entity
phase and coherence explanation remain unproven. No test-guided tuning followed.
This bounded null for attribution does not close the mechanism.

Frozen seed13 setup: public synthetic relation task,192/48/48 records, one
dim16 transformer layer,2heads, FF32, no dropout; Adam .003,100 updates,
batch16 except shared first single-case update. Quantum/dephased/classical have
4643 trainable parameters; disabled/random have3555. Same backbone initialization,
examples, targets and update schedules. Checkpoint selection uses validation
Macro-F1, not test. Full metrics/per-class counts are in the companion JSON.

Execution: authorized3080Ti development server's existing py310 environment,
torch2.7.1, CPU2threads; CUDA was unavailable. No formal training launched.
The portable shell launcher completed all five conditions.

Verification:11 focused tests passed, including existing relation-model test.
Checks cover exact disabled baseline equivalence, Hermiticity/unitarity,
normalization, one-step Q=D null, gauge invariance, loop flux, masks,
finite-difference gradients, entity-swap response, original QK distinctness,
full tensor reader/tamper checks and bitwise interrupted/resumed optimizer state.
The shared sample-trace.v2 validator passed all five traces with complete and
strict gates. The owning semantic validator decoded every full-value artifact,
checked source hashes, metadata coverage, arbitrary coordinate readout including
complex components, and numeric embedding/context lineage.

Case capture:10 records per condition,50 total: first sample ID in each split
at initial/best/final plus first real training update.1940 full numeric artifacts
were retained, including embeddings/QKV, all propagation steps, scores/context,
hidden/pooled/classifier states, predictions/loss and gradients. Each case is
bound to actual optimizer update and model-state hash. Snapshots are replay,
not mislabeled training. Raw runs, weights and per-record data are not in Git.

Only latest and best checkpoint filenames are retained. Exhaustive reader
validation originally took252.50seconds with repeated JSON reparsing; bounded
metadata caching reduced the same focused suite to21.15seconds. Actual value
hash checks remain on every read. Training math was not changed by this cache.

Next recommended step: prepare the formal single-seed runner with original-QK
training control, stronger entity permutation, real-data/GPU preflight and real
portal tensor-slice acceptance. A toy pass permits this preparation under the
project's gate. It does not authorize running full formal data on our server.

Evidence: [design](../q_ecpa_toy_design.md), [review](../q_ecpa_review.md),
[aggregate report](q_ecpa_toy_preflight.json), code-bound config source digest
`99a5f1dcda59e4b84e5af46524f268b4ec0c8fb3277f8bd4388af9347f935bb2`.
Raw artifact root: `/home/Q-Attention/q-ecpa-toy/runs/ecpa_toy/20261017_seed13`.
Formal original-QK training and actual portal browser projection have not been
performed; complete toy numeric capture does not conceal those remaining gates.
