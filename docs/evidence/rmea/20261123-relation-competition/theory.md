# Cross-relation competition: task hypothesis, not L5 theory

Native q_rt=|<phi_r,psi_t>|^2+epsilon>0 and alpha_rt=q_rt/sum_t q_rt.
The new parameter-free transform is G_t=sum_s q_st, c_rt=q_rt/G_t,
alpha'_rt=c_rt/sum_t c_rt. All40classes, including null, enter G_t; no
labels/types/cues enter this transform. Bias, utilities, preparation, effects,
EPS and entity-role inputs remain original. All class logits can change.

Hypothesis: discounting tokens receiving broad cross-class support may improve
entity-pair relation evidence versus null. Broad support is not established to
mean semantic ambiguity; the transform can damage correct evidence. Native
correct/null controls are therefore retained, not only five supported misses.

Known source-specific identities (not new theorems): with native alpha and
w_t=1/G_t, z'_r-z_r=Cov_alpha_r(u_rt,w_t)/E_alpha_r(w_t).
For L=sum_r ell_r z'_r, define D_rt=ell_r alpha'_rt(u_rt-sum_j alpha'_rj u_rj).
Then dL/dlog(q_rt)=D_rt-(q_rt/G_t)sum_s D_st. Native has only D_rt. This
demonstrates an actual changed forward map and cross-class learning path, not
POVM completion by a global scalar (which would cancel in each native row).
Neither sign of a covariance nor coupled gradients guarantees CE/Macro gain.

For normalized states, F_r=|phi_r><phi_r|+epsilon I and S=sum_r F_r>0.
E_r=S^-1/2 F_r S^-1/2 form a POVM. On
chi_t=S^1/2 psi_t/sqrt(<psi_t,S psi_t>), Born(E_r,chi_t)=q_rt/G_t.
This is a known frame/square-root measurement and state-filter interpretation,
not a new POVM/PGM theorem. General nonunitary state filtering, its success
probability, gate and readout costs would need accounting for hardware claims;
no shallow-circuit implementation, NISQ benefit or quantum speedup is claimed.

Sinkformers (Sander et al., AISTATS2022, official original PDFpp1,3 scope-read)
already use alternating row/column normalization for classical attention.
The present rectangular40xT one-column/one-row pass is an instance of known
matrix scaling; it is not converged doubly stochastic attention, especially
when40!=T. Equal-input classical normalized affinities are essential in any
future task-utility study; population_competitive only tests removal of phase
interference with matched quantum parameterization, not strongest classical
parity. No current near-neighbor novelty clearance or L5 nomination.

Current execution request: fixed32exposed TRAIN cached-state probe only,
0training/new word encoding/VALID/TEST. Covariance/gradient/padding/native/POVM
checks plus ALLfour outcome strata. Report real repairs/damages, all5supported
misses' cue mass and margins. No sample/power/exponent/temperature/threshold
search. Frozen prediction changes are descriptive, not independent utility.
A negative result closes this exact frozen intervention, not all training;
trainable comparison requires a separate fixed-budget admission/real-update
fixture through the stable runner. Do not infer a mechanism from error counts.

Structurally different from stopped target-row oracle and shared single-word
attenuation: common nonconstant all-token factor is computed without labels
from allrelation effects; it changes every normalized row jointly. Different
from closed spectrum/memory/scattering/relative-return models: no changed
effects, signed value store, token walk or return circuit. Difference alone
does not establish useful novelty.
