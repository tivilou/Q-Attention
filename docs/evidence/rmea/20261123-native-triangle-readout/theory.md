# Direct native cyclic-phase readout: bounded feasibility, not new invariant

For normalized native relation vector phi_r and marked-valid token vectors psi,
let P_r=phi_r phi_r*, S=mean_subject psi psi*, O=mean_object psi psi*.
The added coordinates are a=phi_r* S phi_r, b=phi_r* O phi_r,
c=Im(phi_r* S O phi_r), m=abs(c). Candidate uses(a,b,m,c);
matched control uses(a,b,m,sign(a-b)m). Both add 40x4 coefficients to the same
frozen native logits, with no intercept. Empty span zeros all four coordinates.

For mixed spans c is exactly the mean over ordered subject/object token pairs
of Im(<phi_r|psi_s><psi_s|psi_o><psi_o|phi_r>). This is established Bargmann /
commutator mathematics, overlapping this project's round6 derivation and stopped
scattering work. The distinct question here is direct decision observation,
retaining the native global effect and avoiding the previous normalized-attention
readout cancellation; it is not a revival or parameter rescue of either old fit.

Swapping stored S/O slots preserves m, swaps a/b and reverses c. Whole native
complex conjugation is implemented by negating prepare Linear RZ output rows /
biases and effect raw RZ parameters. RY and CNOT are real and tanh odd, so psi
and phi conjugate, native logits/probabilities are unchanged, c reverses, and
the other coordinates stay fixed. A learned coefficient absorbs global sign;
there is no absolute semantic orientation or new expressivity theorem. Stored
slot exchange is distinct from re-encoding swapped raw text.

Both tiny-head objectives are convex, and the strictly positive .001 squared
coefficient penalty makes the parameter objective strictly convex. Feature RMS
uses TRAIN only; no centering or VALID tuning. The gradient-infinity convergence
gate is checked independently after a single max300-iteration L-BFGS-B call per
arm; numerical failure stops unresolved. The solver is not trained across seeds
or rescued with a changed optimizer/budget. The feasibility gate is a joint CE
and positive39Macro improvement versus BOTH native and matched control.

Only the new contraction, conjugation and tiny-head gradient need focused
fixtures. No native training, optimizer state, checkpoint/resume, scheduler,
exporter or COS implementation is changed. Existing accepted lifecycle is reused.
All fixed raw inputs and native checkpoint already have pinned COS dependencies.
This short experiment makes no new cloud writes. Reconstructible intermediate
features are private GPU cache; code/config/compact result goes to Git.

A pass is exposed-VALID single-checkpoint incremental utility only, and would
justify separately freezing an independent study. It is not original-fresh5 L4
inheritance, independent confirmation, near-neighbor novelty or L5 acceptance.
