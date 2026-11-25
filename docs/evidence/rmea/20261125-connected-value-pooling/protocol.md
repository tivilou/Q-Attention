# Connected value pooling: fixed small real-data single-seed screen

Observed constraints: mean-state flow helped native Q at fixed small budget but
lost to population and Gaussian heads. Earlier scalar cross-side gating failed
full qualification. Do not reopen those exact profiles. New task hypothesis:
first-moment value readout can discard whether two learned cues coincide at the
same contextual word. The frequency and relevance of this issue are unmeasured;
the frozen contextual encoder or end-to-end first-moment router may already
represent it. This is an exploratory test, not an established failure cause.

Keep each router's attention alpha_ri, existing64-dimensional signed values,
linear decoder and bias. Let mu_r=sum_i alpha_ri v_i. For32 fixed adjacent
channel pairs compute C_rj=sum_i alpha_ri v_i,2j v_i,2j+1 - mu_r,2j mu_r,2j+1.
Return original native logit + sum_j c_rj C_rj. c has1280 parameters and starts
at zero, so native initial logits are unchanged. This is one router with an
extended value readout, not an independent classical classifier residual or
a Gaussian/Q ensemble. All five new arms register82308 scalars; counts do not
prove equal effective capacity.

Five new fits, all from matched native initialization:
1. quantum_covariance: original native Q router plus connected products.
2. gaussian_covariance: strongest existing anisotropic Gaussian base plus C.
3. quantum_mean_product: same added parameters but feature mu_even*mu_odd.
4. gaussian_mean_product: identical mean-product control with Gaussian router.
5. dephased_covariance: native population endpoint plus the identical C.
The Gaussian is exp(-sum gamma_d(k_d-q_d)^2/24), gamma alternating3-sqrt2,
3+sqrt2. Its original EPS1e-6 and masked row normalization are retained.
Reuse nativeQ/nativeD from gen90 and Gaussian base from gen85 with the exact
same cohort, initialization method, order, budget, source and optimizer.
No baseline refits, new task fields, backbone updates or new seeds.

At fixed alpha/values the two uniform two-word sets [(1,1),(-1,-1)] and
[(1,-1),(-1,1)] share zero means/marginals but C=+1/-1. A decoder using only
means/products of means cannot separate that pair with those fixed quantities.
This does not separate end-to-end trainable model classes. C vanishes for one
sampled word and is invariant to value-channel constant shifts. Equivalently,
C=.5 sum_ij alpha_i alpha_j (v_i,a-v_j,a)(v_i,b-v_j,b). These are known covariance
identities and second-order pooling tools, not a new T or a quantum resource.
Nearest work: Bilinear CNN Models, arXiv1504.07889; exact reading scope recorded
separately. Gaussian success counts as classical readout evidence, not Q credit.

Fixed original TRAIN-only8192fit/4096assessment cohort8fb8393b, all40labels,
no exact sentence-group overlap. Seed13/1024updates/b128/micro4, FP32/complex64,
TF32off, original AdamW lr.001/wd.01/betas.9,.999/eps1e-8/clip1. Reuse accepted
screen50ce858f/host238f7e68 byte-for-byte. All five fits finish before assessment,
final update only, all4096/all40. Two allocated GPUs use the stable ready queue.
Each arm900sec, campaign3600sec. No VALID/TEST evaluated.

Primary promotion: Qcov all40Macro strictly exceeds all other four new arms
and all three reused native controls (seven comparisons). CE, complete40class
and Gaussian covariance-v-product difference are descriptive. A passing screen
permits frozen full single-seed design only, not L3/T. Otherwise defer this exact
channel-pair/readout/init/1024 profile; no pairs/scale/seed/cohort/budget/metric/
control rescue or optional replication. Not entire-family impossibility.

Affected checks only: independent pair-sum covariance and double-draw mean
product/gradients at nonzero c, NumPy Gaussian endpoint, zero-c native parity,
padding/singleton/translation, cloned actual8-row update in all five arms.
One bounded exact-code review. No core/checkpoint/resume/COS lifecycle matrix.
Short restartable study keeps existing durable inputs and local best/latest;
zero new COS. Task arrays/models stay GPU-side; Git carries compact reports/code.
All historical results and old scopedL4 remain; newL3/L4-T NOT_ACCEPTED.
