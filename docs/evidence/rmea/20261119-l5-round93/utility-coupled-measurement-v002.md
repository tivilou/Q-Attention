# R93 v002: utility-coupled entropy-proximal measurement

Status: **PROVISIONAL_FOR_FALSIFICATION**, not implemented, not empirically
validated, novelty unresolved, L5 NOT_ACCEPTED. This is a new forward mechanism,
not an interpretation or promotion of the old model. No stopped family reopens.

## Evidence and motivation

R81 has a mixed ECQP screen, not a new advantage. R84 shows that complete
classical-head gains do not transfer uniformly by swapping routing into a fixed
quantum readout. It does not identify coadaptation as the unique cause. R92
localizes the R91 native replay discrepancy to historical CE (~7.85e-9); four
current API/gradient-flag variants match and 34 old partial logits match. It
accepts a diagnostic only, not a completed Case Study or historical full logits.

The hypothesis is that a *sample's actual signed utility* should participate in
the measurement that allocates its token attention. This is not a claim that
the original head ignores utility during training: it receives its gradients.
The proposed difference is an explicit, input-dependent utility-to-measurement
path in inference, with fixed weights and no true label access.

## Exact real-arithmetic correspondence at zero update

Let d=8, valid token projectors rho_t=|psi_t><psi_t|, unit query projector
P_r=|phi_r><phi_r|, and epsilon=1e-6 as in the original executed source.
For every class independently define

    E0_r = (P_r + epsilon I)/(1+d epsilon),
    S = sum_t rho_t,
    U_r = sum_t u_rt rho_t,
    f0_r = tr(E0_r U_r)/tr(E0_r S).

E0 is a strictly positive density matrix, hence a valid binary effect (not a
joint 40-outcome POVM). Since tr(rho_t)=1, its normalized Born probabilities are
exactly (|<phi_r|psi_t>|^2+epsilon)/sum_s(p_rs+epsilon). Thus f0 plus the old
bias equals the original logit in exact arithmetic. This folds the *existing*
epsilon into the effect; it does not add a depolarization hyperparameter.
Bitwise equivalence of a future implementation is untested and not implied.

## Proposed forward rule

Set A_r=U_r-f0_r S, N_r=sqrt(1+tr(A_r^2)), B_r=A_r/N_r. The additive
constant is fixed at 1 in the existing utility units, not selected from data.
For a prospectively fixed eta=1 (no coefficient search), set

    E1_r = exp(log E0_r + eta B_r) / tr(exp(log E0_r + eta B_r)),
    alpha1_rt = tr(E1_r rho_t)/tr(E1_r S),
    z1_r = sum_t alpha1_rt u_rt + bias_r.

There is no second addition of epsilon to these probabilities. E1 is full-rank
and trace one in exact arithmetic; S can be singular but tr(E1 S)>0 for any
nonempty valid sequence. Padding is excluded before all sums. eta=0 reduces to
the original rule analytically. Fixed eta=1 is a first discriminating screen
choice, not a tuned optimum or an entitlement to rescue failed results.

The mechanism adds no learned parameters, but changes the function class and
computational cost. It needs 40 small Hermitian matrix exponential evaluations,
not just the old pure-state circuit. The binary effect can in principle be
realized by a dilation/mixed measurement; no old two-layer circuit identity,
hardware speedup, free state preparation or cheap readout is claimed. An exact
complex classical implementation is a required reference, not a rejection.

## Proposition: a limited forward guarantee, not task utility

For any eta>0 the updated own-class aggregate satisfies f1_r>=f0_r.
With quantum relative entropy D(E||E0)=tr E(log E-log E0), Gibbs variational
optimality gives

    eta tr(E1 B)-D(E1||E0)
       >= eta tr(E0 B)-D(E0||E0) = 0,

because tr(E0 A)=0. Therefore

    f1-f0 = N tr(E1 B)/tr(E1 S)
           >= N D(E1||E0)/(eta tr(E1 S)) >= 0.

N is strictly positive and smooth in every Hermitian matrix entry, including
A=0. The zero matrix gives B=0 without a branch. For A!=0 the inequality is strict: equality of the unique Gibbs optimizer with
E0 would imply B is scalar; tr(E0 B)=0 would then imply B=0. This is a direct
application of a known variational principle, **not a claimed new theorem**.
The original MEG paper is real-symmetric and discusses convex-loss learning;
the Hermitian linear Gibbs argument here is explicitly rederived. Its regret
bounds do not automatically transfer to a normalized signed readout, AdamW
training or sample-dependent matrices.

Adding a classwise constant c to every u_rt leaves A, N, E1 and attention
unchanged and shifts f by c. Unlike v1, positive utility-rescaling invariance
is intentionally dropped: the fixed smooth scale changes relative update size. Simultaneous unitary conjugation of every rho_t
and P_r conjugates the effect and leaves all probabilities unchanged. Individual
token global phases are irrelevant. These properties distinguish the proposal
from coherent token-amplitude erasure and from role/background whitening.

## Explicit task-loss counterexample (must retain)

Take two orthogonal states |+>=(|0>+|a>)/sqrt(2),
|->=(|0>-|a>)/sqrt(2), query P=|0><0|, both original probabilities 1/2.
Let true-class utilities be (+0.1,-0.1), a wrong class (+1,-1), and every other
class utilities zero and bias -10. True and wrong biases are zero. Both classes
have A=c X and N=sqrt(1+2c^2), with c=0.1 or 1 respectively. Let
b=0.5 log((1+epsilon)/epsilon)>0 and beta_c=c/sqrt(1+2c^2). In this
two-dimensional block the updated attention contrast is

    k(beta) = beta/sqrt(b^2+beta^2) * tanh(sqrt(b^2+beta^2)).

Its derivative is (b^2/w^3)tanh(w)+(beta^2/w^2)sech(w)^2>0 for
w=sqrt(b^2+beta^2). Thus 0<k_t=k(beta_.1)<k_w=k(beta_1). Both old
aggregates were zero; the true and wrong updated aggregates are .1 k_t and k_w.
The true CE changes from log(2+38 exp(-10)) to

    log(1+exp(k_w-.1 k_t)+38 exp(-10-.1 k_t)),

which is strictly worse. The increase in the first exponential is at least
k_w-.1 k_t>=.9 k_w, whereas the decrease in the last term is at most
3.8 exp(-10) k_t<.9 k_w. The positive weights outside this block cancel
from its attention normalization; no pure-state logarithm is taken.

This is a valid d=8 state/readout-interface witness. Original finite circuit
realization can be obtained using the R82 first-wire +/-pi/2 preparation and
common two CNOT-ring layers, with the query prepared from |0>. A future canary
must verify the exact source construction rather than infer it from this text.
There is no assertion of a natural RoBERTa input image or a trained trajectory.

Thus the own-logit guarantee is not a CE, accuracy, Macro-F1, calibration,
semantic-necessity or quantum-advantage guarantee. Uniform self-amplification
may be harmful. A task-utility screen and strong classical controls are required.

## Discriminating controls and predictions

1. **Original eta=0 / exact complex reference.** Verify analytic correspondence,
   masks, finite values and gradients before any training. Preserve real versus
   bitwise equality separately. Existing historical gates do not change.
2. **Simplex utility tilt**, beta_t proportional to alpha0_t exp(u_t/M), with
   fixed centered scale M=max_t|u_t-f0| (identity if M=0). This uses the same
   utility access and no new parameters; ordinary utility-aware tilting is a
   serious competitor. Equal learned-parameter count is not compute matching.
3. **Query-block pinching:** replace B by P B P+(I-P)B(I-P), using the same full
   A normalization N, to isolate cross-query coherences. The Q subspace is not
   split into arbitrary eigenvectors. This is a control, not a revived stopped
   dephasing claim.
4. **Wrong-utility alignment intervention (effect path only):** for each sample,
   order all valid token indices by SHA256("r93-utility-v2:"+sample_id+":"+index).
   Map ascending valid destinations to this sorted source order; if it is the
   identity and n>1, rotate sources once. This input-ID-only permutation is
   frozen before outputs, shared by all 40 classes, and never moves padding.
   n=1 is explicitly an unchanged control. It is not a donor-sample experiment.
   Set construction utilities to u'_rt=u_r,pi(t), compute U', f0', A', N' and E'
   from that path only, then use **recipient original u_rt** in the final
   readout sum alpha(E')_rt u_rt+bias_r. Keep rho, S, query and mask unchanged.
   This isolates within-sample token/utility alignment at effect construction,
   not general semantic necessity. Swapping utilities in both paths is excluded.
   Preserve every intervention result, including harms and unchanged cases.


For the two-state witness, block pinching gives B_pinched=0 and no change, while
the full operator changes attention; the simplex control also changes attention.
Therefore the witness discriminates the operator from its pinched control,
**not quantum from all classical functions**. Generic classical density-matrix
calculation reproduces it exactly.

Before training, one fixed CPU-only canary must check identity, positivity,
normalization, shift/conjugation covariance, smooth utility-rescaling response,
finite gradients at zero and nearby points, the two-state
witness and the harmful CE witness. No canary has been implemented or executed.
Only a subsequently reviewed/scheduler-admitted frozen bounded real-data screen
may test utility; the old R81 prefix is development only. No optional seeds,
epochs or coefficient search. Full formal runs remain collaborator-only.

## What could justify continuation or closure

The current recommendation is to implement the canary, not claim this is the
selected formal method. A trivial identity, unsupported source correspondence,
nonfinite gradients or failure against its own exact reference blocks execution.
A valid short screen with no signal closes that *budgeted route*, not all
utility-coupled quantum mechanisms. Any observed gain must be compared with
simplex tilt and pinching before attributing a role to noncommuting measurement.
Full formal and independent evidence, substantive near-neighbor distinction and
a stronger same-core theory remain necessary for L5. The generic Gibbs lemma
alone is insufficient.


## Differentiation contract and v1 findings

v1 remains immutable and its first independent review had two must-fix items;
its nominal PASS_WITH_RISK was not treated as cleared execution approval.
The smooth Frobenius scale above changes the mechanism before any experiment,
drops positive scaling invariance explicitly, and is not coefficient tuning.

Compute the initial logarithm analytically:

    log E0 = log((1+epsilon)/(1+d epsilon)) P
             + log(epsilon/(1+d epsilon)) (I-P).

No eigendecomposition of the sevenfold-degenerate E0 is needed. Compute the
matrix exponential as a differentiable matrix function, not by differentiating
eigenvectors. U, f0, A, N and E1 all remain in the autograd graph; no detach or
straight-through estimator is proposed. In exact arithmetic N>=1 and E1>0,
so these maps are smooth on normalized finite states and nonempty masks. This
does not promise finite floating-point gradients at arbitrary extreme inputs.

The first canary must use complex128 and a separate reference implementation,
test exact-zero and near-zero A, repeated eigenvalues, constant utilities,
identical states, one token, singular S, masks and smooth-point finite
differences. Before every future task screen, freeze finite-input diagnostics
and stop on nonfinite values; no unstated clipping or tolerance repair. A
pinched control shares full N and differs in update magnitude as well as
off-block geometry; it is not by itself uniquely causal attribution.
