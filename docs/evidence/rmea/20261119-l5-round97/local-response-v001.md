# R97 v001: local response, query-block geometry and class competition

Status: primary derivation for independent mathematical review. No new numerical
run, mechanism change, novelty acceptance or L5 promotion. This note applies to
the implemented R93-v002/R94 map, not to the original accepted scoped-L4 method.

## 1. Fixed-input setting and known mathematical ingredients

Fix one nonempty valid token set and one class, including its states, utilities,
query and trained parameters. Let dimension d=8, normalized token states psi_t,
rho_t=psi_t psi_t*, normalized query phi, P=phi phi*, and Q=I-P. For epsilon>0,

    E0 = (P + epsilon I)/(1+d epsilon) = e_plus P + e_minus Q,
    e_plus = (1+epsilon)/(1+d epsilon),
    e_minus = epsilon/(1+d epsilon).

Use the existing implementation's epsilon=1e-6; epsilon below is symbolic for
analysis, not an experimental tuning permission. Define

    S = sum_t rho_t,  U = sum_t u_t rho_t,
    D = tr(E0 S) > 0,
    f0 = tr(E0 U)/D,
    A = U - f0 S,
    N = sqrt(1 + ||A||_F^2),  B = A/N.

All utilities are real and fixed when differentiating eta. The implemented map is

    E_eta = exp(log(E0) + eta B)/tr exp(log(E0) + eta B),
    alpha_t(eta) = tr(E_eta rho_t)/tr(E_eta S),
    f(eta) = sum_t alpha_t(eta) u_t.

The source evaluates eta=1, adds the unchanged class bias, then uses multiclass
cross-entropy. There is no second epsilon. Dependence of A/N on trained parameters
remains differentiable during joint training; it is simply held fixed along this
one-dimensional eta path. Nothing here replaces the training gradient.

The logarithmic-mean superoperator is established mathematics. Petz and Ghinea,
*Introduction to quantum Fisher information*, arXiv:1008.2417v1, section1.2,
pp7–9, Lemma1.1 and Example3/equation13, describe its eigenbasis coefficients
and integral form. The Fréchet derivative of exp and the softmax-loss chain rule
are also known ingredients. We claim no novelty for these objects or identities.

Define

    L(X) = integral_0^1 E0^(1-s) X E0^s ds,
    ell = (e_plus-e_minus)/(log(e_plus)-log(e_minus))
        = 1/[(1+d epsilon) log((1+epsilon)/epsilon)].

By the two eigenvalue blocks,

    L(X) = e_plus PXP + e_minus QXQ + ell(PXQ+QXP).              (1)

This is the covariance-side map. Its inverse, rather than L itself, enters the
BKM Fisher metric on tangent perturbations. Avoid conflating the two conventions.

## 2. Source-specific local response decomposition

**Proposition1.** At eta=0,

    G := f'(0) = tr[A L(A)]/(N D)
      = [e_plus ||PAP||_F^2 + e_minus ||QAQ||_F^2
         + 2 ell ||PAQ||_F^2]/(N D) >= 0.                    (2)

Proof. For finite matrices, differentiating the exponential series and integrating
the corresponding beta coefficients gives

    d exp(H+eta B)/d eta at0
      = integral_0^1 exp((1-s)H) B exp(sH) ds.

With H=log E0 this is L(A)/N. The global exponential trace normalization cancels
in f(eta)=tr(exp(H+eta B)U)/tr(exp(H+eta B)S). Quotient differentiation yields
tr[L(A)(U-f0S)]/(ND). A is Hermitian; its four P/Q blocks are Hilbert–Schmidt
orthogonal, with equal squared norms for PAQ and QAP. Equation1 gives equation2.
Alternatively tr(E0 A)=0 also makes the derivative of tr exp vanish at0.

**Proposition2.** For the actual pinching control, replace the generator by
Pi(A)/N, where Pi(A)=PAP+QAQ and N is the *same full A norm*. Keep the same
original U in the final readout. Then

    G_pinched = [e_plus ||PAP||_F^2 + e_minus ||QAQ||_F^2]/(N D),
    G - G_pinched = 2 ell ||PAQ||_F^2/(N D) >= 0.             (3)

Proof. Quotient differentiation gives tr[A L(Pi(A))]/(ND). Equation1 and block
orthogonality remove exactly the cross blocks. Recomputing N after pinching,
pinching the final readout or changing token states would test a different map.

This is a first-derivative identity, not finite-eta full-versus-pinched task gain,
nor evidence that quantum hardware or a uniquely quantum representation is needed.

## 3. The small-epsilon disparity has a conditional, precise scope

Let tokens, utilities and P be fixed as epsilon decreases. Assume additionally
F=tr(P S)>0. Let A_star=U-[tr(PU)/F]S and N_star=sqrt(1+||A_star||_F^2).

**Proposition3.** G_pinched=O(epsilon). If C=||P A_star Q||_F^2>0, then

    (G-G_pinched) log(1/epsilon) -> 2C/(N_star F),

so the cross-block response is Theta(1/log(1/epsilon)) and dominates the
O(epsilon) pinched response, while both responses tend to zero.

Proof. Centering gives

    tr(P A) = -[epsilon/(1+epsilon)] tr(Q A).

P has rank1, hence ||PAP||_F^2=|tr(PA)|^2=O(epsilon^2).
Since F>0, f0 approaches tr(PU)/F at rate O(epsilon), so A=A_star+O(epsilon),
N approaches N_star and D approaches F. The Q-block term in equation3 is
O(epsilon). Also ell log(1/epsilon) approaches1. Substitute into equation3.

The condition F>0 matters. If F=0, D also shrinks with epsilon; the stated
asymptotic conclusion does not follow. No uniform lower bound on F is established
for our dataset. The proposition holds for fixed representations, not for a
sequence of separately retrained models. It describes derivatives at0; observed
finite-eta attention-L1 ratios in R95 are not tests of this asymptotic formula.

## 4. Classification needs a signed response advantage

For every class r, form its own P_r, U_r, A_r, N_r and G_r as above. Consider the
common eta path z_r(eta)=f_r(eta)+b_r with all parameters fixed. At eta=0 let
p=softmax(z(0)), and let y be the example's label used only for this analysis.

**Proposition4.** The exact local CE response is

    CE'(0) = sum_r p_r G_r - G_y.                            (4)

Consequently local CE strictly decreases iff G_y exceeds the probability-weighted
class response. Every G_r may be nonnegative while CE rises. A pairwise correct
margin z_y-z_k initially increases iff G_y>G_k. These statements say nothing
directly about the discontinuous Macro-F1 metric.

Proof. Differentiate logsumexp(z(eta))-z_y(eta). Bias is constant, so z'_r(0)=G_r.
Differentiate each pairwise margin separately. No extra prediction-time label
access or label-dependent measurement is introduced by this diagnostic identity.

Write G_r=G_diag,r+G_cross,r using equation3. Full and pinched paths share the same
eta0 logits and probabilities. Therefore

    CE'_full(0)-CE'_pinched(0)
      = sum_r p_r G_cross,r - G_cross,y
      = -H_cross,                                         (5)
    H_cross := G_cross,y - sum_r p_r G_cross,r.

Unlike the own-score gap in equation3, H_cross has no guaranteed sign. This is
the relevant signed competition quantity; cross-block activity alone is not
discriminative alignment. The diagonal contribution has an analogous definition.

## 5. Empirical interpretation and falsifiable next measurement

R95 found positive own-class finite gains for all972x40 entries, yet worse Macro
and CE after the direct fixed-checkpoint update. R96 joint training produced a
positive Macro/CE point comparison against native; simplex had slightly higher
Macro but worse CE. Those observed results motivate this note. Neither dataset
was independent of its development, and neither result validates equations2–5
empirically or demonstrates that training caused better H_cross alignment.

Before any new model execution, freeze a separate bounded diagnostic contract:

1. Algebra canary: compare equation2 against autograd and symmetric finite
   differences at eta0 on predeclared fixtures, including a harmful-CE case.
   Compare equation3 using the same full N. Reject a numerical mismatch rather
   than changing tolerances after observing outputs.
2. Model analysis: use exact R96 native/full/simplex final weights, authenticate
   data and source, fix sample set and missing-class accounting before execution.
   At each model's eta0 evaluate signed G/H and local CE derivatives. These eta0
   baselines differ across separately trained models; they are not the models'
   actual eta1 deployed logits. Check the actual eta1 CE separately if needed.
3. Distinguishing prediction: the full/pinched local CE derivative difference
   must equal -H_cross, including examples with opposite H_cross signs when
   present. This tests the implementation and the decomposition, not statistical
   novelty or training causality. Absence of both signs is a descriptive outcome.
4. The substantive hypothesis remains untested: joint optimization improves
   useful signed class competition on appropriate independent data, rather than
   merely increasing all class scores. Reserve independent empirical tests for a
   prospectively fixed design; no outcome-based case selection or repeated
   seed/epoch/epsilon search. New full formal runs remain collaborator-only.

## Contribution ceiling

This note specializes established matrix-exponential/BKM geometry to the actual
normalized RMEA utility path, identifies an epsilon-dependent control mismatch,
and relates it to an exact signed class-competition quantity. These may be useful
research tools. Substantive closest-work novelty, a new nontrivial task theorem,
finite-eta1 benefit, trained causal attribution and independent confirmation are
all unresolved. Accepted scoped L4 is unchanged; L5 NOT_ACCEPTED.

Primary source: https://arxiv.org/abs/1008.2417v1. Read pp7–9 for Lemma1.1,
equation12 and Example3/equation13; title/authors/version checked on p1. The page9
equations were visually inspected. Other sections, supplements and cited originals
were not reviewed for this note. PDF SHA256
fb28de5e1c7725cfe1d51b85cae5d47ec97c0de10b74bc06cd41896a681c0845.
