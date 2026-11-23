# Null-reference inverse filtering: a prospective error-linked screen

The original pinned native head makes 478 errors on the exposed 4000item TRAIN
assessment partition;418 cross the no_relation boundary. A single shared class0
offset has label-aware accuracy headroom only12net errors, whereas the frozen
top2pool contains380current errors. These descriptive counts motivate testing
input-dependent evidence. They do not identify nuisance tokens or bound Macro.
See native-error-headroom-20261123; no independent result is implied.

Keep the native psi_t, phi_r, EPS=1e-6, utilities and decoder. Let alpha_0t be
the unchanged native no_relation attention, including EPS and valid-token masks.
Define N_x=sum_t alpha_0t psi_t psi_t^dagger and R_x=N_x+I/8. For r>0, replace
the effect vector by v_r=R_x^-1 phi_r normalized to unit Euclidean norm. Keep
relation0 native. Then compute original Born+EPS normalized token attention and
original signed-utility pooling. No new parameter or classical residual predictor.
The control uses diag(N_x)+I/8, retaining coherent native reference selection.

N is an uncentered second moment, not a centered statistical covariance or a
certified nuisance state. Its spectrum lies in[0,1]; R has eigenvalues[1/8,9/8]
and condition number at most9. Inverse vectors have norm>=8/9 for unit phi.
If N=I/8, all normalized filtered effects and attention equal native exactly.
An effect in a single eigenspace of N also stays unchanged. A constant N can be
absorbed into static pure effect directions, subject to native circuit reachability.

For a hypothetical signal vector phi and nuisance second moment R, the direction
R^-1phi maximizes |v^dagger phi|^2/(v^dagger Rv). Compared to phi, its gain is
(phi^dagger Rphi)(phi^dagger R^-1phi)>=1. This is standard diagonally loaded
adaptive-beamforming/Rayleigh mathematics, not a new L5 theorem. The actual
classifier averages signed utilities, so this surrogate implies no CE/Macro gain.

The normalized Born score is |phi^dagger R^-1psi|^2/(phi^dagger R^-2phi).
Dropping that denominator while adding the SAME EPS generally changes attention.
The code normalizes effects explicitly. Joint unitary changes of coordinates
preserve the full operation; the diagonal control is coordinate-dependent.

At fixed prepared states there need not be absorption into fixed native effects.
Take e0,e1 token states, phi0=(e0+e1)/sqrt2 and phir=(e0-e1)/sqrt2. With one
token of each type, N=diag(1/2,1/2,0,...); duplicating e0 changes it to
diag(2/3,1/3,0,...). The relative weights of an individual e0 and e1 token change
under filtering, while any fixed native effect's per-token attention ratio stays
fixed. This is a state-port example, not a learned-text or all-encoder separation.

A counterexample prevents overstating filtering: take |+>,|-> tokens,
phi0=|+>, phir=e0, positive utility1on|+> and0on|->; competing logits fixed.
Native positive attention is equal. The full filter suppresses |+>, lowers the
true positive logit and increases CE; the diagonal reference leaves e0 unchanged.
This holds with native EPS. Source fixtures check the matrix example, isotropic
limit, independent dense forward/gradient contraction, padding, exact native
class0 at fixed weights, and a positive-logit gradient through reference to phi0.

Elliptical Attention2406.13770v2 selectedpp1-6 already uses a diagonal Mahalanobis
metric. DAMCHA2610.13261v1 selectedpp1-4 already constructs input-dependent full
cross-head metrics. Generic metric adaptation is prior art. Capon1969 DOI
10.1109/PROC.1969.7278 has metadata confirmation only, not original-text reading.
The precise null-Born reference composition has not been precedence-cleared;
its component identities cannot establish novelty. The source/counterexample
supports a discriminating screen, not L5 acceptance or quantum exclusivity.

One prospective screen starts all3arms from the SAME pinned fulltrained native
checkpoint with fresh identical optimizer state,128updates, same batch32/order/
AdamW and all81028parameters trainable. Fit4192/assess4000 exposed TRAIN only;
allfits finish before any final-state assessment. Use existing two-card lanes,
microbatch backward/metrics/checkpoint helpers and host wrapper. No new encoder,
full formal run, periodic COS or resume claim. Joint gates unchanged: lower full
CE than both, paired exact-sentence bootstrap lower.025>0both, Macro>=both.
Report no_relation boundary transitions descriptively, never for selection.
Failure closes this exact screen without lambda/init/budget/control/seed/split
rescue; a pass buys independent-study preparation only. Original L4 unchanged.
