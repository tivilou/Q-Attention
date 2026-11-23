# Frozen two-group diagnostic in the unchanged original measurement circuit

Question: can a calibration-only common descent direction improve BOTH
assessment label groups and their worst-group CE derivative relative to simple
equal weighting? Keep all480 native effect coordinates, includingclass0; the
checkpoint, original circuit, states, utilities and bias stay unchanged.
All8192 exposedTRAIN rows and exact-sentence4192/4000 split stay included.
Labels0 versus1..39 are the two groups; no outcome-selected subgroups define
the optimization. No steps, optimizer, VALID/TEST, full-training promotion orCOS.

Reuse the previous sealed meanCE calibration/assessment gradients, ordinary
unit direction, logits and perrow logit derivatives from e30ea312... privateNPZ.
Calibrate g0 by a new native forward/backward on ALL2764calibration label0 rows;
recover g+ by g_mean=(2764*g0+1428*g+)/4192. Select directions from calibration
only and seal them BEFORE assessment gradients/derivatives. Compute assessment
g0 on2623label0 rows and recoverg+ from cached4000mean gradient/1377positives.
Check group gradient inner products against fresh perrow JVP mean on BOTH groups
and partitions, including the cached ordinary direction. All native logits and
argmax must match the completed immutable reference. This reuses accepted
calculation instead of rerunning the completed ordinary-direction experiment.

Three directions have Euclidean norm1 in the same native coordinates:
frequency=-unit(g_mean), balanced=-unit((g0+g+)/2), common=-unit(g*), where
g*=lambda*g0+(1-lambda)*g+ and
lambda=clip((||g+||²-<g0,g+>)/||g0-g+||²,0,1).
The last is the known MGDA minimum-norm convex gradient, not new quantum theory.
If the denominator is<=1e-24*max(1,||g0||²,||g+||²), common is unresolved.
If any candidate gradient norm<=1e-12, its direction is unresolved; do not
replace it. Synthetic opposite/equal/zero cases verify these conventions.
If common is defined and nonzero, projection optimality gives
<g0,d_common>,<g+,d_common> <= -||g*|| on CALIBRATION. This does not guarantee
assessment transfer, strict decrease for finite changes or39-class Macro.

The existing native circuit JVP and masked Born+EPS normalization derivative
are reused. Only balanced/common need new logit JVPs; ordinary is read from
its pinned output. One fixed h=1e-5 central difference on phi and first8 rows,
tolerance1e-7; no step-size search or lifecycle matrix. Per-group JVP versus
gradient-inner-product error must be<1e-9. Private outputs stay onGPU.

Primary on the fixed4000assessment rows: W=max(D L_A0,D L_A+).
Require W_common < -tau AND W_balanced-W_common >tau;
tau=1e-8*max(1,||g_A0||,||g_A+||). Invalid directions give UNRESOLVED; any failed
gate closes this exact profile with no group, loss, metric, normalization,
direction, split or seed rescue. Tables/error margins are descriptive.
No independent utility, finite error repair, Macro improvement, phase necessity,
novelty orL5 follows from a derivative result, even if both gates pass.

Prior art scope: Sener/Koltun arXiv1810.04650v2 pp1-4, Eq3/4. This is a
problem-driven diagnostic using standard multiobjective mathematics; novelty
and a nontrivial same-core theoretical contribution remain separate open work.
