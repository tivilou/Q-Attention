# Close this filter; measure routing versus value headroom before another model

The exact loaded-inverse null-reference warmstart128 screen is scientifically
negative:5/6gates fail, full CE.496154>.418078native and Macro.590671<.612598.
Full-versus-native errors rise by133. Native continuation itself loses to the
unchanged starting checkpoint on this exposed TRAIN partition. Preserve that
limitation; it neither rescues the filter nor rejects all training regimes.
No lambda, learning-rate, initialization, budget, split, normalizer, control or
seed rescue; no independent/full-study promotion or L5.

The next step is a bounded diagnostic of the UNCHANGED pinned native checkpoint,
not another trained replacement. Retain its prepared states, utilities, decoder
and bias, and ask which errors are fixable by routing alone. All8192TRAIN rows
and the same4192/4000partition stay included; do not select a favorable class.

For valid tokens rho_t=psi_t psi_t^dagger and fixed native utilityu_rt, define
S=mean_t rho_t+EPS I and B_r=mean_t(u_rt rho_t)+EPS mean_t(u_rt) I.
Then the source-exact native logit is
z_r=b_r+(phi_r^dagger B_r phi_r)/(phi_r^dagger S phi_r), for unitphi_r.
The EPS term in BOTH matrices is necessary. S is positive definite.

If arbitrary pure effect directions may be chosen independently for every
example/relation, the relaxed logit interval is bias plus the extreme eigenvalues
of S^-1/2 B_r S^-1/2. Compare those intervals with the looser range obtained by
arbitrary convex token attention: bias+[min_t u_rt,max_t u_rt]. This is a
label-aware pointwise relaxation, not a shared trained model, native finite-circuit
realizability, independent utility or Macro-optimality claim. Known generalized
Rayleigh/numerical-range tools are not L5 novelty.

For a target y, a positive margin upper_y-max_(r!=y)lower_r demonstrates
classification feasibility within that relaxed per-example family; a strictly
negative certified upper bound rules out routing-only correction under its
fixed states/values/bias assumptions. Handle class-ID ties and numerical
uncertainty explicitly. A near-zero floating result is unresolved, never an
impossibility certificate. Do not infer that globally shared effects attain the
oracle or that training the value/encoder cannot repair such errors.

One concrete next action: freeze and independently review this diagnostic's
exact algebra, scope and numerical uncertainty handling before its bounded
native-forward pass. Use accepted runtime and one existing allocated GPU,
unchanged checkpoint/features, no optimizer, VALID/TEST or COS. This is not yet
an implemented or approved experiment; actual source-specific fixtures and
review are required. Existing aggregate predictions alone do not contain the
per-token utilities and states needed to compute these bounds.
