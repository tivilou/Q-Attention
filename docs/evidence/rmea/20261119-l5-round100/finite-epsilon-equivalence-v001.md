# R100 v001: finite-epsilon rotation, exact query-gradient identity and screening consequence

Status: new primary derivation for independent mathematical review. Exploratory
only; no numerical experiment, implementation, novelty clearance or L5 acceptance.
R99 v001 and its review remain immutable. This note changes the prospective
comparison rather than retuning its epsilon, coefficient, seed or training budget.

## 1. Fixed setting and scope

Let n>=1 normalized token states psi_t and a normalized query phi belong to C^d.
Real utilities u_t are fixed during the one-step intervention. Write * for the
conjugate transpose, P=phi phi*, Q=I-P, and

    S=sum_t psi_t psi_t*, U=sum_t u_t psi_t psi_t*, epsilon>0,
    E0=(P+epsilon I)/(1+d epsilon),
    f0=tr(E0 U)/tr(E0 S), A=U-f0 S, N=sqrt(1+||A||_F^2),
    w=Q A phi, s=||w||, K=[A,P]/N.

The R99 candidate at eta=1 is E_R=exp(K) E0 exp(-K). Nothing below changes
the deployed epsilon=1e-6. All statements concern this exact fixed-input map.
The utilities used to construct A are the same utilities used in final readout.
They do not assert label knowledge at inference, joint-training convergence,
class-level cross-entropy improvement or held-out Macro improvement.

## 2. Exact finite-epsilon classical expression

For any normalized q, define the ordinary floor-regularized attention score

    f_e(q) = [q*Uq+epsilon tr(U)]/[q*Sq+epsilon tr(S)]
           = sum_t u_t [|q*psi_t|^2+epsilon]
                       / sum_t [|q*psi_t|^2+epsilon].              (1)

Equation (1) is exactly the effect readout for
E(q)=(q q*+epsilon I)/(1+d epsilon); it is not an asymptotic approximation.
Let Z=phi*S phi+epsilon tr(S)>0 and a=phi*A phi. With the real inner product
Re(x*y), the sphere gradient at phi is

    g = grad_sphere f_e(phi) = 2 Q A phi/Z = 2w/Z.                (2)

Proof: the ambient real gradient of the quotient is 2A phi/Z. Projecting
onto the unit-sphere tangent space subtracts 2a phi/Z. The resulting g is
also complex-horizontal, phi*g=0, since A is Hermitian and a is real.

For arbitrary fixed eta>=0, set alpha=eta Z/(2N). Then

    exp(eta K) phi = Exp_sphere_phi(alpha g)
                  = cos(eta s/N) phi + sin(eta s/N) w/s,         (3)

with the continuous value phi when s=0. Indeed K=(w phi*-phi w*)/N;
K acts as a two-dimensional rotation on span(phi,w) and is zero off it.
Its induced effect is exactly

    exp(eta K) E0 exp(-eta K)
       = [q_eta q_eta*+epsilon I]/(1+d epsilon).                 (4)

Thus the R99 map is EXACTLY one input-dependent step of ordinary sphere
gradient ascent on floor-regularized attention for every epsilon>0 and eta.
The step length alpha is determined by the frozen recipe, not tuned here.
This strengthens R99's stated pure-limit correspondence to a finite-epsilon
functional identity. No newly invented quantum or optimization primitive is
established. Literature priority for the complete attention prescription is
still a separate question; classical equivalence alone does not reject its
ordinary task utility or demote accepted historical evidence.

## 3. Own-score strict ascent at the actual finite epsilon

**Proposition.** For the fixed eta=1 and every epsilon>0, if s>0 then

    f_e(exp(K)phi)>f_e(phi).                                    (5)

If s=0 the score is unchanged. No numerical tolerance or epsilon sweep is
needed for this algebraic statement.

Proof. Put v=w/s, theta=s/N and c=v*A v (real). Centering gives

    a+epsilon tr(A)=0.

Since phi,v are orthonormal and phi*A v=s, the Frobenius norm obeys

    ||A||_F^2 >= a^2+c^2+2s^2,
    0<theta<1/sqrt(2), |c-a|<=sqrt(2)||A||_F<sqrt(2)N.

For q=cos(theta)phi+sin(theta)v, the exact quotient difference has numerator

    q*Aq+epsilon tr(A)
      = q*Aq-a
      = sin(theta)[2s cos(theta)+(c-a)sin(theta)].               (6)

Using sin(theta)<=theta, cos(theta)>=1-theta^2/2>3/4, the bracket is strictly
greater than s[3/2-sqrt(2)]>0. Its denominator
q*Sq+epsilon tr(S) is positive because S>=0, tr(S)=n>=1 and epsilon>0.
This proves (5). If s=0 then [A,P]=0 and q=phi.

This is a conservative normalized-gradient ascent property. It removes
R99's finite-epsilon own-score uncertainty, but supplies no classification
guarantee: different classes receive different beneficial score changes.
R99's two-class counterexample remains valid at finite epsilon and increases
the true-label CE despite increasing both class scores.

## 4. Consequence for code and the next scientific comparison

Do not schedule rotation and its exactly matched sphere-gradient expression
as two independent task-utility conditions. They are the same function when
the same utilities, normalization N, step alpha and epsilon are used.
A future implementation may compare these as a bounded algebra/gradient
parity test, not as independent scientific evidence.

For joint training, detaching the computed gradient, alpha, utilities or
normalization would change the graph. A correct matched implementation must
differentiate through the same functions (analytic expression or appropriate
higher-order autograd), with the continuous sin(theta)/s limit at s=0.
No forward/backward numerical parity is claimed here; no R99 implementation
has run. A different retraction or unmatched step size is not this identity.

A useful future task screen would compare genuinely different maps: native,
R96 Gibbs, and ONE rotation/query-gradient map under frozen common training
conditions. Existing R96 development evidence does not confirm the new map.
This note does not supply a launch contract or override R98's configuration
writer-serialization admission block. No extra seed/epoch/epsilon/coefficient
search is authorized; all stopped branches and accepted scoped L4 are unchanged.

The new result narrows what needs testing. It does not show that the new map
is independently useful, substantively novel, or L5. L5 remains NOT_ACCEPTED.
