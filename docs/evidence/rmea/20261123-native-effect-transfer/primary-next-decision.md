# Next: test simultaneous group improvement inside the original head

Two completed diagnostics changed the decision: fixed-value pointwise Born
routing has471/478errorwitnesses, and the original480effectcoordinates have
a calibration-selected direction with assessmentCE derivative−.116523.
Yet zero-labelgroup derivative−.258383 coexists with positive-labelgroup
+.153702. This is a tradeoff along ONE direction, not proof of incompatible
gradients, finite Macro harm or the cause of every historical negative.

Keep the ORIGINAL quantum architecture, native coordinates, checkpoint and
states/values/bias fixed. The next prospective bounded question is whether a
single direction selected without assessment labels can improve BOTH label
groups and improve their worst-group derivative over simple equal weighting.

On calibration alone compute row-meanCE gradients g0 and g+ for true label0
and labels1..39, using every row. Include all480effectcoordinates/class0.
Compare three unit directions, with no parameter update or step-size selection:
1. row-frequency-weighted meanCE (reuse the sealed completed result);
2. equally weighted groupmeanCE, d_bal=-unit((g0+g+)/2);
3. minimum-norm convex gradient g*=lambda*g0+(1-lambda)*g+,
   lambda=clip((||g+||²-<g0,g+>)/||g0-g+||²,0,1), d_common=-unit(g*).

Degenerate/zero directions are unresolved, not replaced. This is the standard
two-objective minimum-norm gradient rule, not a novel quantum theorem. Projection
optimality implies <g_i,d_common> <= -||g*|| for BOTH calibration groups when
g*!=0. This does not guarantee transfer, finite updates, Macro or all39positive
relations individually. Explicitly verify the source-specific gradient/JVP
correspondence and degeneration handling with one scoped review before running.

Prospective primary is max(D L_A0,D L_A+) on the fixed4000assessment rows.
Use same native Euclidean unit norm for each direction. Require common direction
strictly negative in BOTH groups and strictly smaller worst-group derivative than
the equal-weight direction, using frozen numerical tolerance1e-8*max(1,gradient
norms). Otherwise close this exact diagnostic; no group/loss/metric/direction/
split/normalization/seed rescue. All other derivative/error/correct tables are
descriptive. Known pointwise oracle bounds or calibration guarantees do not count
as assessment success. No L1/L4/L5 or full-training promotion from derivatives.

This next diagnostic is NOT_IMPLEMENTED/NOT_REVIEWED/NOT_RUN. The completed
one-direction profile is terminal; do not rerun it or modify its reports. Reuse
cached immutable gradients/output identities when possible. No optimizer,
VALID/TEST, fullbaseline repetition, newarchitecture orCOS. Literature novelty
and a nontrivial same-core theory remain separate open work, not a reason to
rename an existing multiobjective optimizer as a new mechanism.

Primary-source check: Sener and Koltun, Multi-Task Learning as Multi-Objective Optimization, arXiv1810.04650v2, pp1-4 scope-read; Sec3.1 Eq3 and Sec3.2 Eq4 contain the minimum-norm rule and two-gradient closed form. This is explicitly prior art, not our novel theorem. Source: https://arxiv.org/abs/1810.04650v2 . Version/hash/read scope in literature-reading.json; no appendix/code/full-results reading claim.
