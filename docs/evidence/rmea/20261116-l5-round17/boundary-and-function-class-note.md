# Distribution identities and restricted output heads

This is an elementary implementation scope check, not a new method or an L5 theorem.

For T valid tokens, unchanged Q-RMEA uses alpha_i=(q_i+epsilon)/(sum_j q_j+T epsilon), epsilon=1e-6, 0<=q_i<=1. Every valid-token weight is strictly positive; invalid masked tokens are separate. Equivalently alpha=softmax(log(q+epsilon)), not softmax(q).

The free-score envelope obeys:

    alpha_i >= epsilon/(T-1+T epsilon)
    alpha_i <= (1+epsilon)/(1+T epsilon)
    max(alpha)/min(alpha) <= (1+epsilon)/epsilon

For independent unconstrained q in [0,1]^T the ratio bound is also sufficient. Choose epsilon/min(alpha)<=c<=(1+epsilon)/max(alpha), then q_i=c alpha_i-epsilon. This describes only a free-score envelope: shared circuit preparation and constrained relation effects may restrict actual reachability further.

Thus the exact sparse-boundary extension in arXiv2608.11173v3 p9 Theorem14 is unavailable for valid tokens in the current epsilon-smoothed implementation. We do not propose removing epsilon to manufacture this property.

Conversely, its pp8-9 pointwise angle/log-score mapping does not equate restricted parameterized heads. On a fixed two-token affine feature curve, existing classical key/query score differences are affine, whereas log(q(x)+epsilon) need not be affine. Equality of unrestricted distributions at each input differs from equality of parameterized functions across inputs.

The Round16 complete-head candidate retains only a strict-margin witness against the exact affine-key comparator, including learned values/readout. It uses known Fourier and exponential-polynomial tools, has no established natural-feature premise or substantive novelty, and a classical nonlinear/cosine scorer can directly reproduce its probability function. Neither this identity nor a successful circuit fixture proves quantum necessity, training advantage or L5.
