# One calibration direction in the original shared measurement circuit

This is a zero-update diagnostic of the SAME native checkpoint and 8192 exposed
TRAIN rows. Existing sentence-group calibration4192/assessment4000, all40 labels,
native states/values/bias and circuit remain unchanged. No independent held-out
claim, optimizer, finite-step fit, step-size selection, VALID/TEST or COS.

Only the native effect tensor theta has requires_grad=true (40x2x3x2=480).
Compute g_C=gradient of row-mean40wayCE on calibration ONLY. If norm<=1e-12,
record no_direction and stop; never substitute another gradient/subset/loss.
Otherwise freeze d=-g_C/||g_C|| in the checkpoint's Euclidean coordinates and
seal it BEFORE any assessment gradient/derivative. Class0 is included.

Compute phi_dot by JVP of the native circuit(phi=U(pi*tanh(theta))|0>).
For amplitude a_rt=<phi_r|psi_t>, p_dot=2Re(conj(a)*<phi_dot|psi_t>),
q_t=(|a_t|²+EPS)*valid_t. Holding utilities/states fixed,
z_dot_r=sum_t p_dot_t*(u_rt-pooled_r)/sum_t q_t.
This differentiates the SOURCE token normalization with fixed EPS=1e-6.
Independently, assessment mean CE derivative is dot(gradient L_A,d).
It must match mean_i sum_r(softmax(z_i)_r-onehot(y_i)_r)*z_dot_ir.
Calibration row derivative must equal -||g_C||. Weights never update.

Primary outcome is sign of assessment mean CE derivative: negative below
tau=1e-8*max(1,||gradient L_A||), positive above tau, otherwise unresolved.
No bootstrap/significance or new evidence-tier claim. Partition and error-type
tables report CE derivative and true-minus-strongest-competitor margin derivative.
For a competitor tie use the maximum logit derivative among EXACTLY tied top
competitors, as required by the one-sided derivative of max. Original error
membership uses the source's lowest-class-ID argmax. Margin changes within
1e-8*max(1,max|z_dot|) are unresolved. Increasing a margin is NOT repairing a
discrete prediction or predicting a finite-step/Macro improvement.

Check prior logits and exact original argmax on ALL8192. Check native phi matches
the independently JVP-evaluated circuit. Native effect JVP and first8 rows' logit
directional derivatives get one fixed central-difference fixture h=1e-5,
error<1e-7, no step-size search. All-row autograd/JVP CE identities have tolerance
1e-9. These are scoped scientific checks; no serial/multiGPU/COS capability
matrix, source changes, new training or model replacement.

A negative assessment derivative supports a locally accessible transfer direction
under this specified learning rule on exposed TRAIN; it does not identify the
quantum phase as necessary or establish independent task gains/L5. Positive or
unresolved derivative closes only THIS direction, leaving other shared/finite
changes open. Standard first-order calculus is not novelty. A perfect-fit shared
effect incompatibility certificate would answer a weaker decision question and
is not selected for this iteration.

Empty descriptive cells have n=0, zero counts and null means, never aborting the
primary result. An exactly zero assessment gradient gives undefined/null cosine
and an unresolved primary sign, while retaining the valid zero derivative.
