# Common-utility routing: fixed TRAIN-only diagnostic

Gen91 Qcov improved native Q but lost to Dcov and Gaussian; each model learned
different value/decoder parameters. The current results do not isolate the
quality of learned word allocation at exactly the same utilities. This bounded
training diagnostic addresses that question; no new quantum operation is selected.

Native logits are b_r + sum_i alpha_ri u_ri, with u_ri=d_r^T value(h_i).
For any two routers sharing utilities/bias, logit difference is exactly
sum_i (alpha_A-alpha_B)u_ri. Constant utilities tie under every normalized router.
These are known weighted-average identities, not T. Uniformly learned utilities
may suppress useful cues, so a negative result only closes this staged profile.

Three fits: quantum, native dephased, original anisotropic Gaussian base.
Each starts from exactly its established native seed13 initialization. For the
first256 updates alpha is uniform; train shared value/decoder/bias only. Router
parameters have exact zero gradient and zero AdamW learning rate. Three copies
repeat the identical deterministic warmup on the same batches; no shared mutable
cross-thread state or new training-core hook. This duplicates512 warmup updates
relative to fitting once, to reuse the unchanged accepted core safely.

At step257 record the shared-state SHA, freeze shared parameters (detach plus
exact zero gradients and zero LR), reset each router's previously zero optimizer
moments/step, and train the native router768 updates. Shared parameters including
weight decay cannot change. Local inspection must prove all three boundary
hashes identical and final shared-state hashes exactly equal their boundaries.
All81028 registered parameters remain present, but only shared68200 or router
12828 are active in the respective stages; nativeD has invariant directions,
so neither equal effective capacity nor equal quantum gate cost is claimed.

Original cohort8fb8393b: TRAIN8192fit/4096assessment, sentence-group-disjoint,
all40labels. Seed13,1024 total updates/arm, b128/micro4, FP32/complex64, TF32off,
original AdamW .001/.01/betas.9,.999/eps1e-8/clip1. Router phase receives batches
257..1024, identical across arms. No changed data, encoder, angle scale, depth,
normalizer, utility features, new seeds or heldout population.

All three fits finish before final assessment; primary all40 Macro-F1. Also
retain all40 confusions/support/F1, CE and learning curves. Practical reference
is fixed gen91 Gcov .512054050292468 at the same total per-path update budget,
not a matched active-parameter or optimizer-path control. Q must strictly beat
both new trained routers AND that practical reference to consider this route
promising. Otherwise STOP_EXACT_COMMON_UTILITY_PROFILE; no warmup/split/seed/
budget/control rescue, no full/extra seeds. A pass is only development guidance,
not new L3 or L4-T.

Stable screen50ce858f/host238f7e68 unchanged. Two newly allocated GPUs and ready
queue;3072 total physical updates, per-arm900s/campaign3600s. Short disposable
screen keeps local best/latest, source+compact results in Git; zero new COS.
No Windows task arrays/model/features. New stage/freezing fixture on cloned real8
examples and one exact bounded code review only; no lifecycle/COS/resume matrix.
If any freeze/cohort/finite correctness fails, mark invalid, not negative science.
