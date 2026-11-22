# Primary next decision: test global measurement mixtures, not another certificate

The interval-order test completed with K=0. All36relations having strictnative
correct positives have overlapping binary interval relaxations; the largest
max_null L-min_positive U is -4.4791046803. Thus this is not a near miss at .05.
No alternate gap, matching order, null contrast, cycle or seed is selected to
rescue this ordering hypothesis. The exact occurrence test also closed for
zero supported keys, not an effect estimate. Both results remain immutable.

The next hypothesis is the PRIMARY's own experimental lead, previously supplied
to Astra for criticism but not recommended/accepted by its advice. There is no
external endorsement or claim of L5 novelty. Rather than more relaxed oracle
certificates, directly test one globally learnable representation change.

## One question

Does a relation represented by a two-component mixed measurement benefit from
the sentence-dependent mixing induced by Born normalization, beyond the same
two pure components averaged after their individual normalizations?

Retain the original native preparation, utilities, decoder and bias, initially
freeze them at the same seed13 checkpoint. Each relation has two original
12-angle pure circuit effects phi_r1/phi_r2, with equal fixed prior weight.
Candidate uses raw score p_rt=(|<phi_r1|psi_t>|²+|<phi_r2|psi_t>|²)/2 and the
original EPS normalization. Matched control uses the mean of the two separately
EPS-normalized attentions. Both have960trainable raw effect parameters, no gate,
intercept or other predictive feature. Both include native when the effects
coincide. This differs from stopped entity-conditioned mixed effects, raw
mention-protection constructions and direct triangle-feature readout.

For Z_l=sum_valid(p_l+EPS) and component pooled scalar z_l excluding bias,
candidate z=(Z_1 z_1+Z_2 z_2)/(Z_1+Z_2), control=(z_1+z_2)/2, so
candidate-control=(Z_1-Z_2)(z_1-z_2)/(2(Z_1+Z_2)). This is an exact KNOWN-type
mixture/covariance identity. It neither predicts correct labels nor establishes
novelty or quantum exclusivity. It identifies the observable changed, permits
exact normalization controls and highlights failure when component mass and
useful value disagree. Generic multihead/mixed-state attention is a direct
collision risk. The earlier known rank-bound theorem is not new justification
for a global trained mixture or for task utility.

## Frozen bounded feasibility

- First8192TRAIN/all19584VALID, same checkpoint/source/features, zero TEST.
  No lexical support selection and no original-fresh5L4 inheritance.
- Native prepare/value/decoder/bias are frozen. Initialize the two raw effects
  at native_effect +/- .1*d_r, where d_r is a unit12vector from NumPy
  default_rng(0) normals in relation/index order. Both arms exactly share this
  initialization. The perturbation avoids an identical-branch stationary split;
  it is fixed before results, not a tuning parameter.
- Two epochs, batch32 =>512updates/arm. Adam lr.001, betas(.9,.999), eps1e-8,
  weight_decay0, no scheduler or gradient clipping. TRAIN order is successive
  default_rng(13) permutations, shared across arms. Float64/complex128, fixed
  final update512, no best-checkpoint selection or validation tuning.
- Evaluate once after final fit. Require candidate VALID CE at least.005 lower
  and positive39Macro at least.003 higher versus BOTH reused native and matched
  post-normalization control. Allfourinequalities mustpass. A numerical defect
  is invalid/unresolved; any scientific miss closes this exact profile with no
  rank/init/step/regularization/cohort/threshold/seed rescue.
- Reuse existing native task forward/circuit/evaluator. For speed, derive fixed
  C_i=sum(rho_t+EPS I) and B_ir=sum u_irt(rho_t+EPS I); component readout is
  (phi*B_ir phi)/(phi*C_i phi)+bias_r. Validate this moment compression against
  actual native masked varying-length inputs, both normalization orders,
  coincident-effect native parity and one finite-gradient real update.
  This reuses the existing Rayleigh reduction as a tool, not new theory.
- One scoped implementation/mechanism review before execution. Reuse accepted
  allocation/training capabilities; no serial/grouped/COS canary matrix. The
  tiny-head arms can be batched on one GPU with shared moments; full follow-up
  independent conditions should use both allocated2080Ti cards via the accepted
  multiGPU runner. No new fleet/cloud/server management.

## Status and claim

NOT IMPLEMENTED/NOT RUN. The next action is to implement/review the isolated
two-effect tiny-head screen and execute only the fixed feasibility profile.
No native baseline retraining. No new mathematical novelty or L5 candidate is
accepted here: a positive incremental result would warrant a separately frozen
full-training study and nearest-method/theory work. Negative short-test weights
need no archive; preserve only code/config and compact results. If positive
weights become an active dependency, retain the minimal effect state with its
exact inputs via the existing persistence adapter, not all moment caches.
Full objective remains the same substantive theory–empirical contribution;
this is a staged attempt to obtain missing task evidence, not a lower goal.
