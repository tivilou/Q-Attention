# Ordered native mass–value statistic: exact scope and prospective diagnostic

For one relation and n valid original tokens, q_i=p_i+EPS>0,
m_i=q_i/mean(q), s=z-b=mean(m*u), v_i=(m_i,m_i(u_i-s)),
and c_i=v_i-mu with mu=(1,0). Then sum c_i=0.
Define A=n^-2 sum_{i<j}det(c_i,c_j), in original token order.
This can be computed using det(prefix(c),c_i) in O(n) per relation.

Expanding every triangle det(v_j-v_i,v_k-v_i) gives coefficient
n-2(j-i) on det(c_i,c_j). But
sum_{i<j}(j-i)det(c_i,c_j)
=sum_j j det(sum_i c_i,c_j)=0. Thus
A=n^-3 sum_{i<j<k}det(v_j-v_i,v_k-v_i).
For B=n^-2 sum_{i<j}det(v_i,v_j), bilinearity also gives
B-A=n^-2 sum_i(2i-n-1)det(mu,c_i), indexing from1.

Under a uniform permutation of FIXED postencoder records, the conditional mean
of A given any named record at any position is zero. Pairs not involving it
cancel by exchange symmetry. Pairs involving it have conditional expected
partner -c/(n-1), whose determinant with c is zero. Hence A is orthogonal in
this permutation-null L2 space to all additive record-by-position functions.
B-A belongs to that space, so it is the additive-position projection of B.
This is a known determinant/sequence-signature/permutation-projection tool,
not a claim of new mathematics or semantic independence on natural sentences.

A is zero for n<3, any two-point contribution support, or uniform m=1.
It changes sign on reversal of fixed records, and is invariant to positive
rescaling of q or common shifts of u. The native contextual encoder already
encodes order; permuting records is NOT permuting text and re-encoding it.
Neither these identities nor an observed A/error correlation proves quantum
necessity, mediation, utility or L5. Nearest-neighbor novelty remains unresolved.

Frozen diagnostic: first8192 stored TRAIN only, all40 CE classes, no updates,
VALID/TEST or checkpoints/COS. Native source/checkpoint/pinned metadata unchanged.
Group by SHA256(JSON(original token strings)); this prevents identical sentences
crossing partitions, but is NOT document/source grouping. Document IDs are not
retained. SHA256(profile-prefix+group-hash) low bit selects calibration/assessment
without labels. Both partitions have already potentially trained the checkpoint;
all estimates/intervals are exploratory TRAIN evidence, not independent support.

For each relation the nuisance design includes an intercept, all40 class-centered
native logits, uncentered-minus-centered position trend P=B-A, four entity-role
moments mean((role-mean(role))*c2), four analogous c1 moments, and log(n).
Fit standardized OLS/pseudoinverse only on calibration, rcond=1e-10. Residualize
intact and broken-pair A separately against their corresponding P/role moments,
using the same native logits. Labels are not used for nuisance fitting.

Broken-pair control uses one deterministic hash-order token permutation per
sample, shared across all40 value rows; masses, positions and original logits
remain fixed. Recompute s and c2 for permuted values. An accidental identity
permutation is rotated once. No multiple draws or favorable permutation choice.

Remove the class-common shift of each residual feature. For both conditions,
normalize each example to unit class-RMS. If either residual is degenerate
(RMS<=1e-12), set BOTH features to zero and retain that example in all counts
and averages. More than1% such examples in either partition fails qualification.
This gives exactly matched per-example perturbation RMS, not just equal scales
on calibration. Raw A is centered/normalized separately for the directional
consistency check and is not used to rescue a failed residual feature.

Use native g=softmax(z)-onehot(y). Each arm independently locks its orientation
sigma=-sign(mean_calibration sum_r g_r F_r), so the broken-pair control gets
the same chance to choose direction. On assessment report intact derivative,
the paired intact-minus-broken derivative, raw-A derivative under the intact
orientation, all40 class contributions and descriptive position-control derivative.
Success requires: both partitions>=1000; >=99% nondegenerate matched rows per
partition; nonzero orientations in BOTH arms; negative assessment mean raw-A derivative;
upper97.5percentile bootstrap endpoints below0 for BOTH intact residual derivative
and paired intact-minus-control. Bootstrap2000 replicates/seed20261123, resampling
whole exact-sentence groups with replacement and weighting by sampled row counts.
These approximate percentile endpoints are not guaranteed-coverage intervals or
confirmatory significance. No seed/sample/control/rcond/threshold rescue.

Failure closes this exact ordered-readout profile. A pass admits plugin preparation
only: no full-run, inherited L4, quantum novelty or L5. Native model parameters,
core runner and lifecycle are unchanged. Exact short-sequence pair/triple/prefix,
position decomposition, conditional permutation, degeneracy, mask/gradient-free
native replay and numerical CE directional-derivative fixtures cover this delta.
Private compact feature tables remain on GPU-server disk/hostRAM; public outputs
are summary.json, fixtures.json and RUN_COMPLETE only.
