# Fixed-native routing headroom, not a new model

All 8192 previously exposed TRAIN rows and the pinned native checkpoint remain
unchanged. The assessment partition has 4000 rows and is exploratory only.
The exact-sentence split and all 40 labels remain fixed; no fitting, VALID or
TEST access. No newly trained parameters, seeds or alternative budgets.

For valid tokens, rho_t=psi_t psi_t†, frozen utility u_rt and native EPS=1e-6,
S=mean(rho)+EPS I and B_r=mean(u_r rho)+EPS mean(u_r) I. For unit effect phi_r,
the native logit is b_r+(phi_r†B_r phi_r)/(phi_r†S phi_r). EPS in BOTH matrices
is necessary, as is averaging over VALID tokens. The implementation checks this
against the unchanged native forward and prior immutable predictions on all rows.
Phi and psi norm errors from floating circuit evaluation are recorded.

The relaxed family allows a different arbitrary unit effect for EVERY example
and relation. Its mathematical closed logit interval is the extremal spectrum
of S^-1/2 B_r S^-1/2 plus b_r. This is standard generalized-Rayleigh theory;
no novelty, circuit reachability or shared-parameter representation claim.
The family is contained in arbitrary convex token attention with interval
b_r+[min u_rt,max u_rt]. Identical prepared states make the quantum range a
singleton even when the convex interval is wide. Orthogonal-state fixtures
have the exact EPS-adjusted analytic extremes, not simply min/max utilities.

We replay both spectral extremizers as explicit unit directions against the
original token weights (|phi†psi_t|²+EPS)/sum_t(...). Joint per-row choices
are allowed because the 40 native binary effects are not one joint POVM.
For label y, set its effect to the upper extremizer and every other effect
to the lower one. Strict positive replay margin >tau is a concrete
LABEL-AWARE POINTWISE witness; it is not a learned predictor or a task gain.
Also report true-row-only improvement against unchanged native competitors,
and all-positive-row intervention with class0 output fixed to native.

Numerical policy is frozen: float64/complex128; tau=1e-7 times
max(1,max|u|+max|bias|) PER EXAMPLE. Require all source/replay/prior/bound nesting
discrepancies <tau. Estimated upper_y-max_other lower < -tau is reported only
as a NEGATIVE NUMERICAL MARGIN, never as a certified outer enclosure or exact
impossibility proof. Near ties and nonpositive witnesses without such negative
margin are unresolved. The source argmax tie policy is lowest class ID; all
feasibility counts require strict positive margins rather than claiming ties.
These conventions neither certify real-arithmetic rounding nor test all
possible encoder/value training. Private witnesses are retained for eight
predeclared first TRAIN examples and reused by an independent verifier.

Interpretation before looking at results: if the convex relaxation already
leaves little usable space, value/representation deserves priority. If convex
space is broad but Born space is narrow, investigate fixed-state routing
geometry. If both are broad, this diagnostic alone cannot distinguish
shared-effect learning, optimization or input-conditioned routing; it is not
permission to declare all such mechanisms promising. No threshold tuning,
sample exclusions, additional checkpoint selection or automatic full run.
