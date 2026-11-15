# Known fusion operator and a limited testable attention hypothesis

The operator is prior partial-swap quantum addition (Audenaert/Datta/Ozols,
arXiv1503.04213v3, eqs3.1 and4.10-4.17). Neither its validity, its ellipse nor
its imaginary triple-product interpretation is claimed as a new theorem.
Fernandes etal arXiv2403.15066v1 is a direct prior invariant-imaginarity witness.
This round tests an attention design; no substantive new theory is accepted.

For one frozen example/relation, let e_t=tr((A+B)rho_t)/2,
d_t=tr((B-A)rho_t)/2, c_t=tr(i[A,B]rho_t)/2. On valid tokens define
a_t=e_t+kappa*d_t+epsilon, A0=sum(a_t), C=sum(c_t). Inside the valid disk,

  alpha_t(lambda)=(a_t+lambda*c_t)/(A0+lambda*C),
  d alpha_t/d lambda=(c_t*A0-a_t*C)/(A0+lambda*C)^2.

Consequently a nonzero commutator or cyclic imaginary trace need not change
normalized attention. If c_t/a_t is constant, every allocation is unchanged.
Even a changing allocation need not change the decision: with fixed token
utility u_t and z0=sum(a_t*u_t)/A0,

  z(lambda)-z(0)=lambda*sum(c_t*(u_t-z0))/(A0+lambda*C).

The classification bias cancels from this difference. These formulas assume
fixed preparation, entity memberships, kappa and readout, real non-clamped
arithmetic and valid-token sums. They do not describe retraining or changes
of frozen contextual features. The implementation records any roundoff
clamps separately. The identity is elementary quotient algebra: recomputing
it from the same selected validation logits would be a correspondence check,
not independent prediction or a nontrivial L5 theory result.

The exploratory prediction was frozen before execution: full fusion must
beat both the ordinary biased entity mixture and original global Q on the
existing exposed validation screen to remain a utility lead. Even a positive
result would not identify quantum phase as necessary: the trainable families
have different gates/parameterizations, all admit exact classical complex
evaluation, and strong classical/dephased controls and independent evidence
remain outstanding. Same-selected-model lambda=0 is a local dependence
diagnostic, not an attribution of total improvement.

The earlier ordered family transforms a global relation effect with two
entity-generated unitaries; it failed its full seed13 comparison. The present
design drops that global effect and fuses mixtures of prepared entity-token
states. This changes the causal link and is not an optional rescue of the
old run. Nonetheless related application novelty remains unverified. The
separate six-qubit joint and spectrum-only screens remain stopped.
