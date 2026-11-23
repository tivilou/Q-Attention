# Conditional context-cancellation boundary — derivation candidate

Scope: mathematical relaxation of the frozen scattering head, not trained-model evidence, implementation authorization, novelty clearance or L5. This follows the first Astra consultation's operator-angle question. The reduction below was derived by the primary after that response. All state directions may vary freely while q/e stay fixed; this generally enlarges the actual circuit/feature family. A necessary impossibility bound transfers to that family, while a relaxed attainable witness need not.

## Exact fixed-head identities

Assume real E/e, eta>0, gamma>0 (the executed contract fixes eta=.1, gamma=1), q_i>0 and sum q_i=1. Thus both terminal damping eta+gamma/2 and the transmission multiplier gamma² are strictly positive. Write f_i=E-e_i+q_i+i(eta+gamma*1_terminal/2). With unit column states, M=I-sum_i(q_i/f_i)psi_i psi_i^H, the off-diagonal terminal entry is

R_so=sqrt(q_s q_o)/(f_s f_o) psi_s^H M^-1 psi_o.

M includes the terminals: this factorization alone is not nonterminal attribution. Remove only their two rank-one terms to define

C=I-sum_{i not s,o} a_i u_i u_i^H, a_i=q_i/(E-e_i+q_i+i eta).

Two-rank determinant cancellation gives

psi_s^H M^-1 psi_o=(psi_s^H C^-1 psi_o) det(C)/det(M).

Every Im(a_i)<0. If Cv=0, taking Im(v^H Cv) gives sum_i[-Im(a_i)]|u_i^H v|²=0, hence all u_i^H v=0, so Cv=v=0. Thus C is invertible; the same argument proves M invertible. Nonzero endpoint factors imply exact zero transmission iff psi_s^H C^-1 psi_o=0. Epsilon in the reported logit prevents negative infinity but does not alter the zero-T condition.

For one fixed C, the largest endpoint overlap allowing this zero is the maximal projective turning sine of C (equivalently C^-1). Indeed, for unit v choose psi_o=Cv/||Cv||; an endpoint psi_s orthogonal to v cancels transmission, and its greatest possible overlap with psi_o is ||P_vperp Cv||/||Cv||. This is the known total-antieigenvalue/operator-angle formulation, not a new definition.

## Joint geometry reduction under fixed coefficients

Let there be m nonterminals and dimension d>=2. For any unit v set t_i=|u_i^H v|² in [0,1]. Then

alpha(t)=1-sum_i a_i t_i,
beta(t)=sum_i |a_i| sqrt(t_i(1-t_i)).

Decomposing u_i u_i^H v=t_i v+w_i gives ||w_i||=sqrt(t_i(1-t_i)). The perpendicular component of Cv has norm at most beta, so its turning sine is at most beta/sqrt(|alpha|²+beta²). This upper bound is simultaneously attained in one common complex two-dimensional plane: take orthonormal v,w and

u_i=sqrt(t_i)v+exp(-i arg(a_i))sqrt(1-t_i)w.

Then sum_i a_i w_i=beta w and Cv=alpha v-beta w. Consequently the sharp relaxed maximal cancellable endpoint overlap is

c_star=max_{t in [0,1]^m} beta(t)/sqrt(|alpha(t)|²+beta(t)²).

This is not yet a novelty claim. Since Im(alpha)=sum_i[-Im(a_i)]t_i, alpha cannot vanish: if any t_i>0 its imaginary part is positive, while t=0 gives alpha=1. Compactness therefore gives c_star<1. For c>c_star exact cancellation is impossible in the relaxed family, and hence in the actual head under the frozen q/e intervention. A maximizer realizes c_star with psi_s=w and psi_o=Cv/||Cv||. Along t=tau*t_star, tau in [0,1], continuity realizes every overlap from0 to c_star; both terminals and all context states remain in the same two-dimensional plane. This relaxation does not enforce joint attainability with the actual Born q or contextual feature/angle map.

For m=0, c_star=0. For m=1, writing d0=|1-a|>0 gives the check

t_star=1/(1+d0), c_star=|a|/(1+|1-a|).

That single-projector case is a normal two-eigenvalue angle calculation and should be treated as prior mathematical territory.

## Convex characterization

Let r_star=max_t beta(t)/|alpha(t)|, so c_star=r_star/sqrt(1+r_star²). For fixed nonnegative r, the condition r<=r_star is a second-order-cone feasibility problem: introduce u_i>=0 with ||(2t_i-1,2u_i)||<=1 and require r*||(1-sum Re(a_i)t_i, -sum Im(a_i)t_i)||<=sum |a_i|u_i.

A stronger two-real-variable dual follows from beta-r|alpha|=min_{|z|<=r}[beta-Re(conj(z)alpha)]. Sion's minimax theorem applies to the compact convex box and complex disk, with a concave function of t and affine function of z. For b_i(z)=Re(conj(z)a_i),

max_{0<=t<=1} [|a_i|sqrt(t(1-t))+b_i t]=(b_i+sqrt(b_i²+|a_i|²))/2.

Thus max_t[beta-r|alpha|]=min_{|z|<=r} F(z), where

F(z)=-Re(z)+1/2 sum_i[b_i(z)+sqrt(b_i(z)²+|a_i|²)].

F is convex on C identified with R². The same sharp r_star equals min{|z|:F(z)<=0}, with attainment. To check the zero boundary, let h(r)=max_t[beta-r|alpha|]. Compactness and min|alpha|>0 give h(r)>0 below r_star, h(r_star)=0 and h(r)<0 above it. The disk minimum of F is attained, so the smallest disk intersecting {F<=0} has exactly radius r_star. For m=0, F(z)=-Re(z) and the minimizing feasible point is0. This reduces an arbitrary number of free context directions to a two-real-variable convex problem with m scalar terms. It is a conditional geometric certificate, not a new physical primitive or a guarantee of task performance.

## Independent check and prior boundary

The owner-selected A6API gpt-6.2-astra/high review in angle-review-result.json returned PASS_WITH_SCOPE and identified one missing stated assumption: positive terminal damping and a nonzero transmission prefactor. The explicit gamma>0 assumption above resolves that omission and matches the already-frozen gamma1 experiment; no model or experiment changed. The review verified the determinant direction, common-plane construction, complete overlap interval, cone feasibility, Sion exchange, closest-point characterization and m0/m1 edges. The consultant read this packet, not the cited original papers or the trained checkpoints.

Paul and Das, arXiv1007.5224v1 pp7-8 already define total antieigenvalues through |x^H T x|/(||Tx||||x||), give the complex-scalar minimax identity in Theorem3.6, and compute a two-dimensional example. Thus the fixed-C operator-angle identification and a minimax argument are established mathematical tools. The conditional optimization over prescribed complex rank-one coefficients and its task relevance remain the narrower research question; this bounded source reading does not certify that they are new.

## Stop conditions

Do not promote this route if it is just an already solved joint operator-angle result with no additional task-relevant restriction, if bounds are vacuous on actual final checkpoints, if practical finite-T/epsilon robustness is absent, or if the already-running frozen utility gate fails. No additional fits, tuned eta, hand-selected text cases or rescue cohorts are authorized by this derivation.
