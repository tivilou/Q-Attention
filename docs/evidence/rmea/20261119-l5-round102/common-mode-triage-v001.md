# R102: common-mode utility and class-dependent pooling

Status: exploratory analytic screening, no implementation or numerical execution.
One prospective mechanism family is identified below. Algebraic duplicates and
variants are not additional idea trials. L5 remains NOT_ACCEPTED.

## Measured starting point

R95's frozen-checkpoint utility update raised every own-class aggregate but
reduced Macro-F1; R96 joint fitting gave a positive full-versus-native development
point, while full Macro remained slightly below simplex. These observations
motivate examining multiclass readout, not assuming positive own-score response
is a classification guarantee. R98 is a separately frozen descriptive diagnostic;
its science and pending one-attempt contract are unchanged.

The actual source is `experiments/rmea_context_geometry/legacy/rmea.py`, SHA256
9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0.
It computes independently normalized class attention alpha_ct, shared values
v_t, decoder rows d_c, utilities u_ct=d_c^T v_t, and

    z_c = sum_t alpha_ct u_ct + beta_c.

States and native routing are independent of the value/decoder tensors. The
R94 measurement update additionally consumes utilities in its construction path.
Padding is excluded throughout; there are C=40 classes, including null.

## Three analytic questions, one surviving candidate family

### Q1. Does per-class scalar utility centering create a new update?

No. For fixed scalar k_c, U'_c=U_c-k_c S and f'_c=f_c-k_c, hence
A'_c=U'_c-f'_c S=A_c. The R94 smooth normalization, Gibbs update and R99
commutator rotation are unchanged. Centering by each class's native expected
utility is already present in A. This is an exact duplicate, not a candidate.

### Q2. Can a token component common to every class affect classification?

Define b_t=(1/C) sum_c u_ct and w_ct=u_ct-b_t. For any fixed normalized
class routes (including epsilon-smoothed Born routes),

    z_c-z_d = beta_c-beta_d
              + alpha_c dot w_c - alpha_d dot w_d
              + (alpha_c-alpha_d) dot b.                 (1)

Thus a common token utility does not generally cancel after class-dependent
pooling. It cancels for all b exactly when alpha_c=alpha_d. For fixed routes,
the sharp bound is

    |(alpha_c-alpha_d) dot b|
       <= (max b-min b)/2 * ||alpha_c-alpha_d||_1.         (2)

Proof: their difference sums to zero, so subtract the midpoint of b's range
and apply the l1/linfinity inequality. Assigning the two endpoints according to
the sign of alpha_c-alpha_d attains the bound. A centered-readout winning margin
larger than this bound against every competitor is sufficient for the original
winner to agree, with strict inequalities avoiding tie conventions. This is a
standard total-variation inequality specialized to this source, not new math.

There is also an exact source-restricted characterization. Under a common
decoder translation d_c -> d_c+g, fixed native routes give

    Delta(z_c-z_d) = g^T(mu_c-mu_d),
    mu_c = sum_t alpha_ct v_t.                            (3)

All decoder translations leave every softmax probability unchanged iff all
mu_c agree. For a Euclidean perturbation ball ||g||<=r, the exact maximum
absolute pair-margin change is r||mu_c-mu_d||. When mu_c!=mu_d and the starting
pair margin is positive m, m/||mu_c-mu_d|| is the infimum translation norm needed
to reach the pair tie (not necessarily the multiclass decision boundary).
Proofs follow directly from the shared affine decoder and Cauchy-Schwarz.
Equations (1)-(3) hold for fixed routes. For the utility-dependent Gibbs/rotation
path, recomputing routing adds terms; do not apply (3) to that full intervention.

This is not the R67 value-affine-bias gauge: a constant token-independent shift
with a compensating class bias preserves logits. Here a common decoder vector
produces a token-varying component whose differently pooled contributions can
change margins. Calling it a nuisance is a hypothesis, not a consequence of
the algebra. It may encode useful evidence through the class routes.

### Q3. Does centering only the construction path remove that component?

Let w_ct=u_ct-b_t. Construct the measurement from w, but read out the original
u. An arbitrary class-common token shift u_ct -> u_ct+h_t leaves w and hence
the constructed measurement unchanged, with states fixed. Nevertheless each
logit changes by alpha_c dot h. Its pair change remains
(alpha_c-alpha_d) dot h, so final predictions need not be invariant.
Construction-only centering is a distinct path intervention, but does not prove
common-mode robustness of the classifier.

## One prospective family: joint class-centered construction and readout

Set dbar=(1/C)sum_c d_c and w_ct=(d_c-dbar)^T v_t. Use w in BOTH the measurement
construction and the final aggregation; retain the original learned class bias.
Keep the same C, source epsilon, normalized positive attention, states and
smooth update scale. No complementary measurement or signed attention map is
introduced. The centering is across all classes at each token, not across tokens.
For fixed states, arbitrary class-common token shifts then leave the entire
candidate exactly unchanged. This is an algebraic invariance of the proposed
utility interface, not a guarantee for semantic/input perturbations.

The same operation is expressible as a centered decoder. Its average row is a
null parameter direction in the candidate, reducing effective decoder dimension
by value_width despite identical stored tensor sizes. It is not an equivalent
reparameterization of the unrestricted native class-dependent-pooling model.
Any benefit could be ordinary regularization rather than measurement updating.

No unconditional CE or Macro benefit follows. A normalized two-token/two-class
witness uses orthogonal token states, complementary query probabilities 3/4 and
1/4, and the unchanged epsilon floor. Let

    r=(3/4+epsilon)/(1+2epsilon),
    s=(1/4+epsilon)/(1+2epsilon),
    alpha_1=(r,s), alpha_2=(s,r),
    w_1=(1,0), w_2=(-1,0), b=(0,4), beta_1=beta_2=0.

Centered margin is r+s=1. Uncentered margin is
1+4(s-r)=1-2/(1+2epsilon)<0 for 0<epsilon<1/2, including 1e-6.
Centering helps label1 but harms label2, and changes their CE in opposite
directions. This is an abstract normalized-state/native-routing witness, not a
claim of an executed original-circuit fixture or the full updated path.

## Minimal prospective discrimination

Before any scientific execution, freeze one joint-fit design with the existing
development split and unchanged optimizer/init/order budget, then obtain its
exact implementation/lifecycle review and scheduler admission. Compare:

1. unchanged native and unchanged R94 full (reuse only exact-contract controls);
2. class-centered native (same readout restriction, original Born routing);
3. class-centered full (one new candidate family, fixed eta1);
4. construction-only centered full if needed to identify the final-readout link.

This is a proposed design, not authorization to add jobs to R98 or reuse R96
numbers as matched controls after a changed execution contract. Store the same
tensor sizes but report effective decoder dimensions separately. Include null
in the uniform class mean; no label-prior weighting or favorable-class selection.
Preflight must check the invariance, all-class denominator, padding, finite
gradients, and exact baseline-source parity. No optional seeds/epochs/centering
weights or post-hoc subset changes.

The discriminating question is whether class-centered full improves the declared
task metric beyond class-centered native and ordinary uncentered full. If only
both centered variants improve similarly, carry decoder restriction forward and
do not attribute it to the measurement update. A fixed-model postfit screen
cannot settle a training-dependent family. Frozen short-screen nulls are
inconclusive unless they reveal a fatal defect or catastrophic degradation.
All new formal full-data runs remain collaborator-only. No L5/theory priority
or quantum-necessity claim follows from an invariance test or this design.

## Source reading and collision scope

Reused the exact local original HTML of Differential Transformer,
arXiv:2410.05258v1, SHA256
f6bdb1bbcd6a32bf983b0632ed4b1458e0fb4021bc1dc32139b5e965a5809121
(373303 bytes). Identity/title/authors/version and introduction, Section2.1,
Eq1-3 and Figure2 were read. The source subtracts two softmax maps with a learned
coefficient and applies per-head normalization. Here the proposed subtraction
is over class utilities/decoder rows while attention stays a probability vector.
That operational difference does not establish novelty. The original paper
already motivates subtraction using common-mode rejection in electrical
engineering. Its LLM gains and gradient claims do not transfer to this head.

Official v1 original: https://arxiv.org/html/2410.05258v1
Current metadata: https://arxiv.org/abs/2410.05258
Current abstract is v2; do not silently relabel the reused v1 reading.
Broader centered-classifier/contrastive-attention priority remains unresolved.
The prior stopped binary-outcome contrast subtracted two token distributions;
this candidate does not reopen its fixed-checkpoint contract or use its failure
as evidence against every centered decoder. No code or empirical result for
this new family has been produced.
