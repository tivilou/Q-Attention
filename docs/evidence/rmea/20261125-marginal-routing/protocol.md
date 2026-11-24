# Two marginal routes for separate signed value coordinates

Status: prospective exposed-TRAIN feasibility study, not L3/T. The native
contextual encoder, three-qubit circuit, 12 input/query angles, affine 64-value
projection, signed decoder and all 81,028 registered scalars stay unchanged.
Hypothesis: imposing one token weighting on every value coordinate can prevent
useful gold-versus-rival combinations. Previous aggregates motivate this question;
they do not establish that this restriction caused the observed errors.

In flat state-vector order reshape Q states to A×B×C=2×2×2; A is the most
significant tensor axis (no assumption about a library's qubit naming). Form AB
and BC reduced density matrices for each valid token and relation. Each overlap
Tr(query_marginal token_marginal) is a separate probability in [0,1]. Add native
EPS=1e-6, mask, and normalize each route over valid tokens independently. The
first route pools even coordinates of value×decoder, the second pools odd
coordinates. Sum both signed pools plus the unchanged class bias. All 40 classes,
including null, use this rule. No labels, external features or classical residual.

Four conditions: quantum_marginal_split; population_marginal_split (diagonal
overlap after coherent preparation/partial trace); quantum_marginal_tied (replace
both normalized maps by their arithmetic average); real12_marginal_split (native
normalized real12 vectors reshaped 2×3×2, same reductions/value split). Same
registered count does not imply same effective capacity or local dimensions.
Population control retains internal coherent preparation. It is not the removal
of all quantum coherence. Neither local measurement is the other's complement.

All four fits: seed13, fixed grouped TRAIN4192 fit / 4000 assessment, 512 updates,
b128, FP32/complex64, TF32off/deterministic/native AdamW lr.001/wd.01,
betas(.9,.999)/eps1e-8/clip1, identical batch order and shared value initialization.
Two GPU lanes, all fits terminal before one final/all40 assessment; no official
VALID/TEST or baseline refit. Reuse four exact same-protocol native Q/real12/
softmax12/RBF12 compact records from heralded-token-prior-20261125.

Seven utility gates: candidate all40 Macro strictly greater than the three new
controls and all four reused references. One mechanism prediction: at the same
fitted candidate parameters, tying both routes lowers mean gold-minus-largest-
rival margin across ALL positive-labeled assessment rows. Recompute the largest
rival separately for each readout. A bounded <=120s forward replay occurs after
all fits within summarize, before RUN_COMPLETE/host return; checkpoint, cohort,
labels and original logits are pinned, maximum original FP32 replay error<=3e-5.
No fitting/selection/extra checkpoint. Main utility and mechanism gate all eight
required. CE, positive/null counts and individual class changes are descriptive.

At fixed parameters, delta_logit = .5 sum_t(alpha0-alpha1)(u0-u1). This permits
outputs beyond one common convex pool, but does not separate arbitrary retrained
classical/shared-route models. Check query000/tokens001,100/orthogonal value
groups as a witness; check GHZ+/- identical marginals as an adverse example.
Partial trace, multi-head/grouped value routing and this identity are known tools,
not an accepted substantive T innovation. No utility guarantee is asserted.

Failure closes this exact marginal choice, coordinate split, normalization and
512 profile, without subset, angle, gain, initialization, seed, budget, metric or
control rescue. Pass only permits complete predeclared qualification preparation.
Does not declare the entire family impossible. All earlier closed profiles and
historical scoped L4 stay unchanged. Short fits are cheap restartable: local
best/latest only, zero COS and no Windows datasets/models/per-example arrays.
