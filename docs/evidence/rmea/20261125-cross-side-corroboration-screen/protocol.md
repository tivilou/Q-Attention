# Cross-side corroboration: small real-data single-seed screen

Observation: complete shared-query necessary-condition checks obstruct 431/3611
native Q TRAIN errors at fixed keys/decoder/rivals, versus155/3070 real12 errors.
They do not identify a cause. Hypothesis: selected subject-side cues need
corroboration from the designated object's context. Contextual frozen embeddings
already contain both marked entities; a missing-binding diagnosis is unproven.

Use raw12 learned keys and the complete retained anisotropic Gaussian scorer,
b_ri=exp[-sum_d gamma_d(k_id-q_rd)^2/24], alternating gamma=3−sqrt2,3+sqrt2.
Partition valid words by smaller absolute continuous distance to the two entity
spans, ties to subject. For subject-side word i, g_i is its mean native3qubit
state fidelity to object-side words; object-side g=1. Empty object-side falls
back to all g=1. Scores b_ri*g_i+nativeEPS, token normalization, signed value and
affine class decoder are unchanged. Seven81028-registered-parameter arms:
quantum gate; raw computational dephasing; unit-self-normalized population gate;
identical Gaussian pair gate; Gaussian alone; native real12; native softmax12.
Same key/value/decoder initialization, input fields, order and optimizer. This is
a new cross-side gate insertion, not a photon/phase/scale retune or reopening the
closed standalone02/Fock or entity-transition profiles. Gaussian base is itself
a coherent-state overlap; multiplying it by a state-mixture Born probability
uses independent registers/projections, no classical additive residual. No
hardware advantage is claimed.

Known tools: quantum fidelity, density means, Gaussian kernels and cross-attention.
Under side-local frozen feature/utility assumptions and fixed rival threshold,
independent token-score threshold numerator is A_a+B_b+c and cannot strictly
separate both matched cells from both mismatches in a2x2 XOR witness. A one-sided
gate can break that restriction; a classical pair gate can too. Contextual
embeddings violate the side-local premise in general. This is a narrow algebraic
motivation, not novel theory, actual natural prevalence, or a quantum separation.
One good corroborator diluted by m object tokens has gate1/m: a failure mode.

New fixed future-screen cohort is sampled only from complete TRAIN, by exact
sentence-token group hashing with fixed prefix/seed, class-stratified8192 fit
and4096 assessment indices. Both must cover all40 labels with no common group;
all counts/indices/hash bindings are frozen before model execution. Natural
class proportions are approximately retained; minimum rare-class coverage
changes them slightly. Old4192/4000 screen had38assessment labels; do not
replace its historical results or reuse its control metrics for this new cohort.
No VALID/TEST prediction or evaluation, and no task arrays or weights go to Windows.

Seed13,1024updates/b128=16epochs over8192 fit; FP32/complex64, deterministic,
TF32off, native AdamW .001/.01/betas.9,.999/eps1e-8/clip1. Final update only,
all7fits finish before assessment; no best/epoch selection by assessment. Same
accepted screenv6 training/evaluation core with a small prospective cohort-loader
extension; default historical path unchanged. No resume/COS capability matrix.
Two allocated GPUs, dynamic independent-arm queue. Each arm600sec, campaign3600sec.
Restartable short screen, existing durable inputs/localbest-latest;0newCOS.

Primary screen promotion: quantum all40Macro strictly exceeds each of6controls.
CE, all40class results, null transitions and paired corrections are descriptive.
All results retained. Failure defers this exact profile without full/multiseed
promotion; it is not a family impossibility and receives no seed/subset/scale/
budget/control rescue. A pass permits a separately frozen full protocol only
after a TRAIN object-summary permutation measurement checks the proposed
context dependence against Gaussian gate; no automatic full launch from exit0.
Full single-seed independent-contribution gates must pass before a complete
predeclared replication roster. No selected seeds or repeated TEST-based search.
Current experiment is exposed TRAIN development evidence, not independentL3/T.

Changed checks only: direct NumPy gate/density/logit/gradient parity, mask and
empty-side endpoints, actual8row cloned update, and new cohort-index/data/group
bindings. One bounded exact-code/mathematical review before fits. Novelty and
deep T review are deferred until utility warrants it; originalL4/history kept.
