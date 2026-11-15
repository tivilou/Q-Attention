# A known measurement interpretation and its transfer boundary

For normalized token states rho_t and epsilon>0, set A_t=rho_t+epsilon I,
S=sum_t A_t. Define a uniform token ensemble with prior 1/T and normalized
states sigma_t=A_t/(1+d epsilon). Its average is
sigma_bar=S/[T(1+d epsilon)]. The pretty-good measurement in Barnum and Knill,
arxiv:quant-ph/0004088v2, Section V p10 Eq37, is exactly

M_t = sigma_bar^(-1/2) [sigma_t/T] sigma_bar^(-1/2)
    = S^(-1/2) A_t S^(-1/2).

The whitened Q-RMEA token effects are therefore the known square-root
measurement of this regularized ensemble. This is a useful identification, not
a claim of novelty. Q-RMEA evaluates those effects on
w=S^(1/2)phi/sqrt(phi^†Sphi), an arbitrary learned, sentence-transformed query.
The near-optimal *ensemble state-discrimination* guarantee is about average
success when the input is drawn from the stated ensemble with its stated
priors. Neither semantic token relevance nor downstream CE follows merely
from evaluating the same measurement on this different query.

A counterexample is two orthogonal token states in d2 with a balanced query
phi=(|0>+|1>)/sqrt(2). The measurement distinguishes the regularized ensemble
with success (1+epsilon)/(1+2epsilon), arbitrarily near one, while Q-RMEA
attention is exactly (.5,.5). Selecting which token is relevant requires
additional assumptions about the query and task. Do not import the PGM
near-optimality result as an existing Re-TACRED utility guarantee.

Similarly, the classical–quantum mixture
rho_TQ=(1/T) sum_t |t><t| tensor rho_t is separable. Conditioning it on a query
effect produces normalized token probabilities, with a direct local hidden-state
description for the token register. That construction alone does not establish
EPR steerability (Uola et al., arxiv:1903.06663v3 p3 Eq2). A coherent purification
is a different construction; its off-diagonal token coherences are not an
observed object in the current head's classical probability pooling.
