# Decoder-conditional interval ordering, prospective and numerical

Native logits z_ir=sum_t alpha_irt u_irt+b_r use relation-specific distributions.
Fix these learned utilities and valid masks. A comparator has any sentencewise
probability vector a_it shared across all40classes, and arbitrary global biases
beta_r. Its r-versus-null contrast is beta_r-beta_0+sum_t a_it d_it(r),
where d_it(r)=u_irt-u_i0t. The attainable interval is exactly
beta_r-beta_0+[L_i(r),U_i(r)], the minimum/maximum token contrast. This is known
convex-hull/interval mathematics, not a new generic separation theorem.

For one gold-r sentence i and gold-null sentence j, suppose native is strictly
correct on both and L_j(r)-U_i(r)>=.05. Every shared pooling then scores the
null sentence's r-versus-null contrast at least .05 higher than the positive
sentence's contrast, independently of arbitrary common bias difference.
Correct comparator classification of i requires contrast>=0; correctness of j
requires contrast<=0. The two are incompatible with this strict ordering.
Thus at least one native-correct decision must be lost. K globally row-disjoint
pairs imply at leastK such lost decisions for every comparator in the family.
Attention may depend on the entire input and even the gold label. The proof
still holds. Cross-class simplex compatibility only restricts the comparator
further; interval relaxation is a conservative certificate.

This is not a bound on net total error or Macro: other rows may improve, and
retraining the value/decoder changes all intervals. Identical useful nonuniform
effects can make native exactly shared-pooling reproducible. Wide token ranges
can destroy the certificate even when class-specific pooling matters. A miss
is not universal dispensability. No quantum/Born exclusivity or originalL4
five-seed inheritance follows from this one seed13 fixed checkpoint.

The new task-linked question is whether enough REAL correct decisions exhibit
this bias-robust obstruction, beyond previous raw-logit matching or synthetic
rank/port observations. It does not add predictive features, optimize an oracle,
change a circuit or rescue a closed witness/phase/occurrence experiment.
All19584VALID rows are fixed before labels/predictions score the event. Match
strictly-correct positives in stored order, select the smallest qualifying null
L with index tie-break, consume null rows globally across all39relations.
Thresholds are K>=196 and >=8relations each with>=5pairs; gap>=.05. Any count
miss closes this exact hypothesis. Invalid numerical replay stops unresolved,
with no drop/reselection. No alternative contrast/cohort/order/gap/seed rescue.

The inequalities use float64 model computations with every selected pair
recomputed from cached token values to1e-8. They are numerical certificates for
stored coefficients, not interval/exact real-circuit certification. The analytic
theorem applies to exact inequalities under its stated fixed-decoder family.
Known interval dominance and classical class-specific attention are direct
novelty risks. A pass is a finite-sample necessity observation only; substantive
L5 novelty and appropriate independent evidence remain to be established.
