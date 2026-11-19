# Cross-fold utility measurement: a provisional selection/evaluation hypothesis

Status: one new exploratory mechanism family for mathematical triage. Not
implemented, numerically executed, empirically beneficial, novelty-cleared or L5.
The ready six-condition centering experiment remains frozen and takes priority.

## 1. Evidence and transfer boundary

R96 jointly fitted full utility measurement improves development Macro and CE
over native, but does not beat simplex on Macro. R98 finds all own-class local
responses positive while the classification-loss response has both signs.
These observations motivate studying selection/evaluation coupling; they do
not establish a noisy-utility model, estimation bias or a causal explanation.
The stopped post-fit insertion is not reopened: any future task comparison
would jointly fit this changed map from the common initialization.

Hasselt's Double Q-learning (NeurIPS2010), Section2.1/2.2, separates selection
from evaluation using independent estimates, and explicitly admits downward
bias for the maximum-value target. The Double DQN original v3, pages1-2, notes
that nonuniform estimation errors can alter preferences, while uniform optimism
may leave decisions unchanged. These are known principles, not new quantum
results or guarantees about NLP. Our folds contain different tokens rather than
two estimates of each identical action. The exact target below is therefore a
clean readout under a random route, not the optimal latent token value.

## 2. A source-preserving map with disjoint construction and readout

Fix normalized token states and the query, E0=(P+epsilon I)/(1+8epsilon),
epsilon=1e-6 and eta=1. Include all40 classes. For each sample split the valid
positions, in their existing order, into alternating nonempty folds A and B.
The partition uses no label, ID, prediction, outcome or learned tensor. A
single valid token takes the native route. There are no new learned parameters,
extra decoder, sampled views, dropout rate, epsilon or tuneable coefficient.

For D in {A,B}, define the existing source quantities on donor fold D:

    S_D=sum_{t in D} rho_t; U_D=sum_{t in D} u_t rho_t;
    d_D=tr(E0 S_D)>0; f_D=tr(E0 U_D)/d_D;
    A_D=U_D-f_D S_D; N_D=sqrt(1+||A_D||_F^2);
    E_D(eta)=exp(log(E0)+eta A_D/N_D)/tr exp(...).

Let m_D=d_D/(d_A+d_B), computed from native states only, with its ordinary
training gradient retained. For a recipient token in B, use donor A:

    alpha_t^cross = m_B tr(E_A rho_t)/tr(E_A S_B), t in B;
    alpha_t^cross = m_A tr(E_B rho_t)/tr(E_B S_A), t in A;
    z_cross=sum_t alpha_t^cross u_t + beta.

This remains a single standalone measurement-based head. All token values enter
the readout; no classical-plus-quantum residual is introduced. Both donor and
recipient states/values remain differentiable during training, with no detach.
There are two classwise matrix exponentials, versus one for unpartitioned full.

Positivity and token normalization hold for finite utilities. At eta=0 the
cross route is exactly the original native route mathematically, because the
native fold mass cancels the recipient denominator. No bitwise numerical parity
is asserted before actual implementation tests. Singleton donors have A_D=0;
singleton recipients cannot redistribute their sole token. Consequently this
alternating two-fold map equals native for every sample with at most3 valid
tokens, at every eta. This is a falsifiable edge case rather than a silent fix.

## 3. Exact separation under an explicit conditional noise assumption

Condition on states, partition, latent clean utilities v and all fixed model
parameters. Suppose u_A=v_A+epsilon_A and u_B=v_B+epsilon_B, where errors have
conditional zero mean and the two error vectors are conditionally independent.
Arbitrary dependence within a fold is allowed. For each recipient fold its
weights depend on the other fold's error and on fixed states, not its own error.
For integrable readouts, conditional expectation therefore gives exactly

    E[sum_t alpha_t^cross(u) epsilon_t]=0,
    E[z_cross(u)-beta] = E[sum_t alpha_t^cross(u) v_t].       (1)

This is unbiasedness for the stated randomly routed clean readout. It is not
unbiasedness for a maximum, the native score, a noiseless-construction score,
a posterior probability or a classification-risk optimum. No CE improvement
follows. Real contextual features and shared learned parameters can correlate
errors across folds, so disjoint positions do not establish the assumption.
For dependent errors the residual is a sum of route/error covariance terms,
which can have either sign. Independence is not claimed for the real dataset.

## 4. Source-specific local cross-covariance geometry

Let M(X)=integral_0^1 E0^(1-s) X E0^s ds, the established positive self-adjoint
logarithmic-mean map. Fix utilities and differentiate eta at0. The exponential
trace normalization cancels inside each recipient quotient, giving

    G_cross = d z_cross/d eta at0
      = tr[A_A M(A_B)] (1/N_A+1/N_B)/(d_A+d_B).             (2)

Its sign is the sign of cross-fold BKM covariance. It is positive for aligned
nonzero residual operators, negative for opposing ones and zero when they are
orthogonal in this inner product. Cauchy-Schwarz gives the exact bound

    |tr[A_A M(A_B)]| <= sqrt(tr[A_A M(A_A)] tr[A_B M(A_B)]).

A compute-matched self-fold control uses E_D to route its own recipient fold D,
with the same fixed m_D, tensors, parameters and two exponentials. Its response
is instead

    G_self = [tr(A_A M(A_A))/N_A + tr(A_B M(A_B))/N_B]
                 /(d_A+d_B) >= 0.                         (3)

There is no uniform cross-over-self dominance. Both remove the original full
map's possibility of steering based purely on between-fold utility means;
that lost information is a possible failure mode, not presumed nuisance.

An additional discriminating local statement fixes eta and expands around
u=0. Let B_t^D=rho_t-alpha_t^{0,D} S_D, where alpha^{0,D} is native attention
normalized within fold D. The cross attention Jacobian has diagonal fold blocks
zero and off-diagonal blocks

    J_cross = eta/(d_A+d_B) [[0,C],[C^T,0]],
    C_st=tr[B_s^A M(B_t^B)].                               (4)

Its nonzero eigenvalues occur as positive/negative singular-value pairs.
The self-fold control instead has positive-semidefinite Gram diagonal blocks.
For bounded zero-mean utility perturbations sigma*epsilon with independent
folds, the leading quadratic expected cross self-selection term vanishes;
the corresponding self-fold term at the stipulated eta=1 is nonnegative and is
strictly positive when the residual perturbation is nonzero with positive
probability. Equation(1) actually removes the cross noise readout at every finite amplitude under the
same conditional independence, not merely to second order. These facts apply
to this interface noise model, not an observed NLP noise process.

## 5. Minimal discriminating path and limits

An eventual synthetic test would check native-at-zero, token normalization,
padding/all40 classes, the <=3-token degeneracy, full gradient lineage,
finite-difference equations(2)/(4), aligned versus opposing covariance, and
independent versus correlated finite noise. Predicted removal of a stochastic
bias is not a classification improvement gate. No numeric execution is admitted
by this proposal; exact code/lifecycle review and shared scheduling remain due.

A future jointly fitted short task screen would compare native, original full,
self_fold and cross_fold under the same existing data/init/order/budget. The
self_fold condition separates cross-fold routing from ordinary partitioning,
native fold masses and doubled compute. Classical matched cross-fold routing
and appropriate independent confirmation would be required before attribution
or L5. The present repeatedly used development cohort cannot supply that
confirmation; no official held-out data or optional rescue seed is requested.

The new causal link is donor/recipient separation. It is structurally distinct
from utility class-centering, query rotation, and stopped post-fit insertion.
Nearest attention and quantum-measurement implementations remain unsearched;
this limited RL transfer reading does not clear novelty. The conditional noise
assumption may be inapplicable, and useful evidence may fail to repeat across
the two folds. These are serious reasons the candidate might fail.

## 6. Completed mathematical triage

One exact owner-selected gpt-6.1-sol/medium consultation completed as
resp_0f82e00557c62877016aff2c80b76087d290caff95ffa761d7: PASS_WITH_RISK,
seven checks passed, zero must-fix. The reviewed mathematical body is unchanged.
For the strict self-noise statement, residual perturbation specifically means
at least one nonzero operator sum_t epsilon_t B_t^D with positive probability;
nonzero token noise alone does not suffice. Conditional mean independence of a
recipient's error given the other fold is sufficient for equation(1).

The provisional mechanism is retained for preparation. Implementation execution,
empirical benefit, novelty and L5 were explicitly not accepted. The public review
record retains every limitation, including shared contextual-error dependence.
