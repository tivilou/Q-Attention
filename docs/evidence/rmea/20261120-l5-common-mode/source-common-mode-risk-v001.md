# Common-mode risk: fixed routes and utility-dependent measurement differ

Status: exact mathematical consultation PASS_WITH_RISK, zero must-fix. No new
numerical execution, training, empirical validation, novelty acceptance or L5.
This refines the existing R102/R103 centering family; it is not another candidate.

## Source and question

The actual map is `experiments/rmea_utility_measurement/head.py`, SHA256
7afde7ef88b60afdf08c87f1a1d812aca33234830d24f2403220af9cc7d6c009.
The centered interface is `class_centered.py`, SHA256
2a214804fe1314e1c6c34c6f069eabeeb260d6cd34582620ce55cbc5b6037974.
Dimension is eight, epsilon is the existing 1e-6, eta is the existing one, and
centering includes all forty classes. No coefficient tuning is proposed.

R103's eight managed synthetic checks passed, including translation invariance.
Those checks do not show that removing the common component improves task risk.
R96/R98 also show that positive own-class utility response does not imply a
classification benefit. The question here is narrower: does an additive,
zero-mean, label-independent common utility justify centering by itself?

## 1. A useful fixed-route result, using established convexity

Write token utilities as u_ct = w_ct + b_t, with sum_c w_ct = 0. Fix the
row-normalized routes A=(alpha_ct), the centered utilities W, bias beta and label
y. Assume E[B | A,W,y] = 0 and the losses below are integrable. Set

    z0_c = sum_t alpha_ct w_ct + beta_c,
    Z(B) = z0 + A B,
    ell_y(z) = log(sum_c exp(z_c)) - z_y.

Then Jensen's inequality gives

    ell_y(z0) <= E_B ell_y(Z(B)).                         (1)

Equality holds exactly when A B is a class-common scalar vector almost surely.
To see this, cross-entropy is strictly convex on logits modulo span{1}; its
Hessian is diag(p)-p p^T with kernel span{1} for every finite logit vector.
Apply strict Jensen to the quotient logits. Equation (1) compares fixed
predictors at a fixed conditional setting, not trained model optima.

For B=xi a with E[xi]=0, the fixed-route directional curvature is

    d^2 ell_y(z0 + xi A a)/d xi^2 at zero
        = Var_{c~p}[(A a)_c] >= 0.                       (2)

These are standard convexity/averaging facts. They establish neither mathematical
novelty nor the premise that the measured data satisfy the conditional mean
assumption. Label independence alone does not imply that assumption.

## 2. Why the full implemented map is outside that argument

For an arbitrary smooth logit path z(xi), the exact chain rule is

    ell_y'' = z'^T [diag(p)-p p^T] z'
              + (p-e_y)^T z''.                          (3)

The first term is nonnegative. The second can have either sign. In the full
measurement update, routing depends on utilities, so z(xi) is not generally
affine. Centering utilities before recomputing that nonlinear route is not the
same operation as taking the mean of the uncentered logits. Thus equation (1)
does not supply a risk guarantee for centered_full versus full.

This is also a literature boundary. Chen, Dobriban and Lee's group-averaging
framework uses an actual estimator/loss average and appropriate distributional
invariance. Its Section 4.1 Lemma 1/Proposition 2 do not identify our nonlinear
centered classifier with that average. Their Section 6 explicitly exposes a
bias-versus-variance issue under approximate invariance. The additive utility
translation group is noncompact, so there is also no uniform Haar probability
over all such translations to insert without further assumptions.

## 3. A source-compatible forty-class counterexample

This construction uses normalized states accepted by the actual measured_attention
interface. It does not establish reachability from an original raw-input circuit,
nor represent an experiment on Re-TACRED or a trained optimum.

Use two valid token states psi_1=e_0 and psi_2=e_1 in C^8. Let a=(1,-1), and give
every class the same utility vector u_c(xi)=xi a. Its class-centered utility is
identically zero. Take the true label y=0, query

    phi_y = (e_0+e_1)/sqrt(2),
    phi_c = e_0 for all c != y.

Keep all forty classes. Set beta_y=0 and beta_c=log(9/39) otherwise. At xi=0 the
true-class softmax probability is p=1/10; the remaining mass is 9/10. Labels do
not enter the forward update. A symmetric random sign for xi is independent of
this fixed label and has zero mean.

For each query, use the original initial effect, not a new epsilon:

    E0 = lambda_plus P + lambda_minus Q,
    lambda_plus = (1+epsilon)/(1+8 epsilon),
    lambda_minus = epsilon/(1+8 epsilon),
    D = tr(E0 S) = (1+2 epsilon)/(1+8 epsilon),
    r = 1/(1+2 epsilon),
    L = log((1+epsilon)/epsilon).

Let U_a = rho_1-rho_2, f_a=tr(E0 U_a)/D and A_a=U_a-f_a S. Under u=xi a, the
actual source has A(xi)=xi A_a and

    update(xi) = xi A_a/sqrt(1+xi^2 ||A_a||_F^2).

Hence update'(0)=A_a. Define the established logarithmic-mean map

    M_E(X)=integral_0^1 E0^(1-s) X E0^s ds,
    K_c=tr[A_a M_E(A_a)]/D.

The quotient derivative of the original exponential update gives

    z_c'(0)=f_a,c,       z_c''(0)=2 eta K_c.              (4)

The smooth normalization contributes no extra term to (4): only update'(0)
enters the derivative of the attention average before multiplying by xi.
The global trace normalization cancels in that average's quotient. This reuses
the R97 Frechet-derivative identity on a different, utility-amplitude path; it
does not substitute the eta derivative unchanged.

For the balanced true-class query, f_a,y=0 and A_a=diag(1,-1) on the token
subspace. Its two query/off-query blocks give

    K_y = 2 / [(1+2 epsilon) L].                         (5)

For every other query, f_a,o=r and A_a commutes with E0, giving

    K_o = 1-r^2 = 4 epsilon(1+epsilon)/(1+2 epsilon)^2.   (6)

Substitution in (3) yields the exact source-specific local curvature

    ell_y''(0) = (1-p) [p r^2 + 2 eta (K_o-K_y)].        (7)

At the unchanged eta=1, epsilon=1e-6 and p=1/10, this is strictly negative.
No numerical evaluation is needed: r^2<1, K_o<1/100000, and L<16. Also
1+2 epsilon<9/8, so K_y>1/9. Therefore the bracket is less than

    1/10 + 1/50000 - 2/9 < 0.

For completeness L<16 follows from exp(16)>(5/2)^16>1000001. Thus, for sufficiently
small positive h, analytic smoothness implies

    [ell_y(z(h))+ell_y(z(-h))]/2
        = ell_y(z(0)) + h^2 ell_y''(0)/2 + o(h^2)
        < ell_y(z(0)).                                  (8)

Both centered_full and centered_native stay at beta for this entire utility
path, because every centered utility is zero. Their CE equals ell_y(z(0)).
Thus removing this zero-mean, label-independent common component strictly
increases the symmetric expected CE compared with the uncentered full map near
zero, at these fixed model parameters. This does not compare retrained optima,
prove an input-domain causal claim, or show that such a component is useful in
the actual dataset.

For fixed native routing, and for construction_only with zero centered
construction, equation (2) instead gives strictly positive curvature
p(1-p)r^2. The direction, source epsilon and bias are identical. This is a
discriminating prediction about nonlinear routing, not a new attention method.

## 4. The conditional boundary is more informative than blanket invariance

For this explicit geometry only, equation (7) changes sign at

    p_star = 2 eta (K_y-K_o)/r^2.                        (9)

The factor is symbolic, not a parameter-search permission. At the fixed source
epsilon and eta it describes when routing curvature overcomes the convex
softmax term in this constructed example. It is not a universal confidence
threshold, an empirical estimate, a calibration rule or a prospective selector.

The existing six-condition developmental training contract remains unchanged.
It is still needed to decide whether centering is useful after fitting, whether
any gain exceeds the ordinary decoder restriction, and whether that gain merits
independent evidence. Equations (7)-(9) are not new task-utility evidence and do
not inherit the original scoped L4.

## 5. Literature and claim ceiling

- Chen, Dobriban and Lee, A Group-Theoretic Framework for Data Augmentation,
  arXiv:1907.10905v4. Read title/authors, Sections 3.1, 3.4, 4.1 and 6.1,
  including Lemmas 1/16 and Propositions 2/17. Known averaging mathematics;
  broad benefits do not transfer without the operation and assumptions.
  Original: https://arxiv.org/html/1907.10905v4
- Zhao, Tachet des Combes, Zhang and Gordon, On Learning Invariant
  Representations for Domain Adaptation, ICML/PMLR97 (2019). The original
  abstract and introductory scope warn that invariant representations alone
  need not ensure useful prediction under conditional/domain shift. This is
  a different setting, not a theorem establishing (7) or evidence about our
  dataset. Original: https://proceedings.mlr.press/v97/zhao19a.html
- Achille and Soatto, arXiv:1706.01350: metadata/abstract discovery only. No
  full-text theorem from that paper is used here.
- The R97 note and its original-source reading identify the Frechet exponential
  derivative/logarithmic mean as established tools. Neither is a new quantum
  principle. Priority for the present source-specific specialization is unresolved.

This note closes an overstrong theoretical justification, while retaining the
existing empirical candidate for its frozen test. No original-circuit numeric
fixture, new optimizer step, COS request or formal handoff was performed.

## 6. Completed review and primary source check

One owner-selected A6API gpt-6.1-sol/medium mathematical consultation completed:
resp_09744625e0ab00b1016aff2730f8c087d1be58cabfde02bc3b, PASS_WITH_RISK, zero
must-fix, ten checks passed. The exact pre-review draft and request hashes are
retained in the private review capsule; the mathematical body above is unchanged.
The public math-review-record.json preserves the accepted claim and all risks.

The consultant did not receive the imported legacy source. Separately, the
primary inspected that source without importing Torch: EPS is literally 1e-6,
and native alpha is proportional to |<phi,psi>|^2+EPS. For normalized states,
tr(E0 rho)=(|<phi,psi>|^2+EPS)/(1+8 EPS), so its common factor cancels and the
stated E0 fixed routes agree algebraically. This is a primary source check,
not a claim of independent review of the omitted file or a new numerical test.
Raw-circuit reachability, an explicit finite h, empirical benefit and novelty
remain unestablished. The frozen six-condition training contract is unchanged.
