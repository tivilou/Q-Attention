# Shared-effect witness: scope and finite-precision certificate

This is an application of known convex duality/tomography, not a novel L5 theorem.
It examines the seven already selected descriptive validation cases, without
training, new samples, test tensors or changing the accepted L4 claim.

Fix normalized token states rho_it in dimension d and an independently chosen
binary effect E for one relation, E >= 0 and tr(E)=1. Effects for different
relations are independent; no joint-POVM sum constraint is imposed. With
epsilon=10^-6, a feasible allocation is

    a*_it = [tr(E rho_it)+epsilon] / [tr(E R_i)+T_i epsilon], R_i=sum_t rho_it.

For any real finite witness coefficients w_it and supplied target a_it, define

    W = sum_it w_it (rho_it - a_it R_i)
    c = epsilon sum_it w_it (T_i a_it - 1), m=tr(W)/d.

If |a*_it-a_it| <= tau, the weighted residual satisfies

    |tr(E W)-c| <= B = tau (1+epsilon) sum_i T_i sum_t |w_it|.

This follows from 0<=tr(E rho_it)<=1, so the positive denominator is at most
T_i(1+epsilon). Also

    |tr(E W)-m| <= sqrt(1-1/d) ||W-mI||_F,

because ||E-I/d||_F^2=tr(E^2)-1/d<=1-1/d. Consequently a strictly positive

    (|c-m|-B)^2 - (1-1/d)||W-mI||_F^2,

with |c-m|>B, certifies that no valid shared effect realizes the supplied target
within the declared coordinate tolerance. The ball bound is only necessary;
failure to obtain this certificate does not establish PSD or pure feasibility.

Witness selection uses an orthonormal traceless Hermitian basis and SVD of the
necessary linear equations A x=b for E=I/d+sum_j x_j F_j. Two candidates are
considered: a residual direction, and the minimum-norm dual direction
U diag(s^-2) U^T b. Rank thresholds affect search only: the final proof works
for any finite w, including an inaccurate SVD result, and does not assume that
the numerically selected w is an exact null vector. All40 relations and all5
heads remain in the screening record. Exactly the strongest screened relation
per head is independently interval-certified; no statement that all relations
are infeasible is permitted.

The saved complex64 psi and phi components, target allocations and selected
float64 w are treated as exact finite binary input numbers. rho is defined by
the exact normalized outer product of saved psi. Directed Decimal intervals
at precision50 enclose all arithmetic used for the final scalar certificate.
The sign test uses squares, avoiding an assumed directed square-root rounding
rule. Intervals cover arithmetic, whereas tau=10^-5 is a separately declared
resolution for matching allocations. This is not a universal float32 error
bound. All original Q allocations must lie within tau of their normalized
saved-state/effect physical reconstruction, verified with intervals. Uniform
is a separate feasible positive control (E=I/d). If either control fails, stop;
do not relax tau after seeing results.

The assay fixes preparation. A donor head generally has a different trained
preparation/value map. A certificate says nothing about feasibility after
changing preparation, quantum necessity, semantic correctness, task utility,
or the cause of the observed swap harm. The selected examples are posthoc and
cannot provide independent task confirmation or L5 acceptance.
