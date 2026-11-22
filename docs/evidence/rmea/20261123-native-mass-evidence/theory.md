# Native acceptance-rate contrast: exploratory missing-readout hypothesis

Motivation: the ordered mass/value statistic has a favorable point derivative but
failed both predeclared bootstrap gates. It is closed; no larger sample, different
orientation/control/seed or trained ordered plugin is admitted. The next hypothesis
uses a different quantity, independent of token order and of classification values.

For the unchanged native state/effect define rho_bar=mean_t|psi_t><psi_t|,
P_r=|phi_r><phi_r|, D_r=Tr(P_r rho_bar)+EPS, and
D0_r=Tr(P_r Delta(rho_bar))+EPS, where Delta removes off-diagonal entries in the
original computational basis. Candidate feature f_r=log(D_r)-log(D0_r).
Both D and D0 are strictly positive. D equals the native normalizer divided by
valid-token count. D0 uses the same learned effects and token populations.

For smoothed density matrices (rho_bar+EPS I)/(1+8EPS) and its dephased counterpart,
f_r is the log likelihood ratio for the BINARY ACCEPT outcome of effect P_r.
It is not the likelihood ratio of class r, a joint40outcome measurement, a full
binary KL divergence or a coherence monotone. No such data-generating class model
is assumed. f is zero when rho_bar is diagonal or P_r is diagonal and is invariant
to global state/effect phase. These are standard density-matrix/likelihood tools,
not a new theorem. The classifier may already encode correlated information in z;
the empirical question is whether this readout supplies incremental error signal.

Keep the original native predictor frozen. First8192 stored TRAIN and the exact
prior sentence-group calibration4192/assessment4000 split are reused, as development
data already exposed to this research and potentially to checkpoint training.
No independent validation is claimed. No parameter update, VALID, TEST or COS.
Read only pinned native inputs and the prior sealed compact diagnostic table.

Control: one fixed hash-ordered cycle deranges all40 relation columns of f. This
preserves each example's multiset of evidence values while breaking its relation
assignment. It is NOT a matched trained classical gate or a quantum-necessity test.
Intact and deranged arms each residualize against the SAME target-relation nuisance
design: intercept, all40 centered native logits, log(D0_r), mean_t u_rt, global
||rho_bar-Delta(rho_bar)||_F, and log(length). Standardization and per-relation OLS
use calibration only/rcond1e-10, reusing the previously reviewed helper.

Class-center and per-example unit-RMS normalize the residuals. If either arm is
degenerate, both are zeroed and the example remains in all averages. Require >=99%
nondegenerate rows in EACH partition. Both arms independently lock orientation on
their calibration CE derivative; either zero orientation fails. Native logits stay
fixed when evaluating all derivatives. Raw f must also give a negative assessment
mean under the intact locked orientation. Whole exact-sentence groups are resampled
2000times with seed20261123; both intact and paired intact-minus-deranged derivative
upper97.5percentile endpoints must be below zero. These are approximate exploratory
bootstrap checks, not guaranteed coverage, confirmatory significance or utility.

All gates are required; otherwise close this exact profile without changing
cohort, controls, scale, threshold, orientation or bootstrap seed. A pass permits
new plugin preparation only. The known mass/logratio identities alone never earn
L5; nearest-neighbor work on quantum classification, postselection and confidence
readout still needs original-source comparison before any novelty promotion.

Only new fixtures: scalar Born/dephased means versus density-matrix forms on fixed
complex states/effects, zero contrast for diagonal states/basis effects, and actual
native logit/normalizer replay. Reuse accepted CE, calibration OLS, matched RMS and
group-bootstrap helpers; do not rerun the ordered area tests or stable lifecycle.
Core training/runner/source unchanged; private compact tables stay on GPU server.
