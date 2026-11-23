# Subject-addressed signed value reading: exploratory trainable screen

The previous ordered-area and acceptance-rate frozen profiles are closed. This
proposal changes the trainable value path, retaining the native81028parameter
flat state, encoder features, three-qubit circuit and40binary effects. It is not
a frozen-logit correction or a new claim about the already accepted scopedL4.

For original subject words S and valid non-subject writes T, normalize unchanged
native Born scores+EPS over S only to beta_rs. Define
Q_r=sum_s beta_rs rho_s, W_r=|T|^-1 sum_t u_rt rho_t,
z_r=b_r+8Tr(Q_rW_r). W is signed Hermitian, not a state. No final normalization
over writes: zero-sum cross-address cancellation is intentional. The diagonal
control replaces the trace by sum_i Q_ii W_ii in the original computational
basis, retaining coherent subject selection. Both arms train preparation,
effects, value, decoder and bias from the same fresh native initialization.
The native short-fit arm is only a same-budget control, not historical baseline
reproduction. All use fixed encoder features and the accepted backward/chunk,
metric and atomic-state helpers without editing those helpers or full runner.

## Conditional ideal retrieval proposition: sharp diagonal MSE >=1/2

Let d=8. Two banks contain d distinct address slots each. In each bank the
payload vector is uniformly sampled from balanced +/-1 vectors; banks are
independent. A query names one of2dslots uniformly. Assume its encoded query
depends on the named address but not the payload draw. Arbitrary trainable
encoders are fixed across draws. The write memory may depend arbitrarily on
all payloads, so the lower bound does not assume a linear write encoder.

For the intact ideal construction encode bank1 as the computational basis and
bank2 as the tensor-Hadamard basis. Each query is its slot's pure state; writes
carry utility2*payload. Equal-basis squared overlaps are Kronecker deltas;
cross-bank squared overlaps are1/d. Hence d/(2d)*2 times the selected payload
is the target; the other bank's balanced contribution vanishes. Every query is
retrieved exactly. This is an ideal state-port construction, not a proof that
the native finite tanh circuit/linear preparer reaches it on actual features.

For any diagonal read, stack predictions on all2dqueries at a fixed payload
draw: yhat=A w+b1. A contains normalized query diagonals, so A1_d=1_(2d).
Thus 1 is already in col(A), and dim(col(A))<=d. The balanced target covariance
is C=d/(d-1)*P_H, where H is the2(d-1)dimensional space of within-bank
contrasts. H is orthogonal to1. If Pi projects onto col(A), at most d-1
dimensions can contribute to Tr(Pi P_H), since one direction is occupied by1.
Even with an arbitrary payload-dependent w, expected per-query squared error
is at least

Tr((I-Pi)C)/(2d) >= [2(d-1)-(d-1)]*[d/(d-1)]/(2d) = 1/2.

This strengthens the consultation's valid but loose3/7 by accounting for the
normalization direction. It is sharp: the same two-basis construction with a
diagonal read retrieves bank1 exactly and returns0on bank2, for MSE1/2.
The fixture enumerates all70^2=4900balanced draws and all16queries, checks the
intact and particular diagonal errors plus the covariance formula. Enumeration
checks the construction; the general bound rests on the stated proof, not
exhaustive testing of encoders. Known covariance projection and MUB tools are
not themselves a new L5 contribution. If a contextual query contains the
answer, A may depend on payload and the lower bound fails. No such locality or
balance is established in Re-TACRED, and MSE separation implies no CE benefit.

## Nearest originals read and collision risk

Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers
with Linear Attention*, arXiv2006.16236v3, selectedpp2-3, already expresses
kernel attention by accumulated key/value features and associative matrix
products. The projector feature map vec(rho) makes the proposed contraction a
known kernel-memory computation. Normalizing over subject words and leaving
the final write sum unnormalized are this task interface's choices, not proof
of a new general attention principle.
Source: https://arxiv.org/abs/2006.16236v3

Schlag, Irie and Schmidhuber, *Linear Transformers Are Secretly Fast Weight
Programmers*, arXiv2102.11174v3, selectedpp2-4, explicitly relates linear
attention to fast-weight memory and discusses finite feature-dimensional
retrieval limitations and earlier tensor-product representation results. This
is a close prior for both mechanism and capacity reasoning. Their selected
pages do not supply the exact balanced-two-basis proposition, but that limited
reading is not a novelty search or precedence clearance. Quantum associative
memory/operator-frame neighbors and earlier cited TPR theorems remain open.
Source: https://arxiv.org/abs/2102.11174v3

No public or private claim of a new memory principle, quantum exclusivity,
classical separation or L5 is admitted by this screen. A classical outer-product
implementation exactly reproduces the intact read. The utility experiment is
still useful because the trainable entity/value interaction has not been
tested in this project's fixed features. Known neighbors do not supply its
task outcome. A positive result would need a more substantive contribution
and independent evidence before L5 consideration.

## Frozen bounded experiment

Use the already exposed first8192TRAIN/exact-sentence split from the pinned
ordered diagnostic:4192fit/4000assessment. No VALID/TEST access or encoder
passes. Fit all head tensors from one seed20261123 native initialization;
AdamW lr.001/weight_decay.01/betas(.9,.999)/eps1e-8/clip1, float64,512updates,
batch32 via the accepted microbatch4 backward. All arms have identical fitting
indices, batch stream and initial state. Finalupdate512 is selected. No
assessment is read for model or checkpoint selection. Local latest is saved
each update and best is training-minibatch-loss retention only; neither chooses
reported weights. Cheap screen has no periodicCOS or automatic retry/resume.

Inside one dualGPUallocation, one process uses two independent device threads:
coherent then native on card0; diagonal on card1. No random operation affects
forward/backward after the precomputed common initialization and batch order.
Both lanes settle before process completion/resource return, including when
one raises. No new scheduler, subprocess worker, full-run recovery or exporter
changes. Host wrapper/permit/ack/release remain unchanged.

All gates required: coherentCE lower than diagonal and native; lower2.5percent
endpoints of2000whole-exact-sentence paired bootstrap comparator-minus-coherent
CE differences positive; coherent positive39Macro >=both. Fixed bootstrap
seed20261123. These are conditional, approximate development checks, not
confirmatory confidence guarantees or independent utility. A pass permits
independent-study preparation only. Otherwise close this exact screen without
seed/split/gain/control/mask/budget rescue; a short-budget miss does not prove
family impossibility. No paid expansion or collaborator-host access.

Focused checks: direct operator versus gathered-overlap forward AND gradient
parity; padding invariance; finite gradients/actual first optimizer state change
per arm; exact synthetic balanced-population fixture. Reuse accepted core
capabilities; no serial/multicard-resume/COS matrix. Public outputs are only
summary.json, fixtures.json and RUN_COMPLETE. Arrays, per-arm raw result logs
and latest/best/final checkpoints stay private; no CaseStudy materialization.
