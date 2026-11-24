# Fixed number-coherent optical routing: prospective complete-data qualification

This is a new frozen source-number-coherence hypothesis, not a retuning of the
closed displaced-|1> profile. Six modes pair the original twelve real projected
coordinates with denominator sqrt(24). Every source mode is fixed
(|0>+|2>)/sqrt(2), with mean photon number one and a different photon-number
distribution/support from |1>. No state, balance, phase, mode, scale or gain search.

For delta=x+iy and s=|delta|^2, the coherent one-mode probability is
exp(-s)[1-s+s^2/4+(x^2-y^2)/sqrt(2)]^2. Full base-number dephasing before
displacement gives rho=(|0><0|+|2><2|)/2 and
Tr(rho D(delta)rho D(delta)^dagger)=exp(-s)[1+(1-2s+s^2/2)^2+s^2]/4.
Dephase both source states before displacement, not the displaced states.
The separate purity-normalized control doubles this latter probability per
mode. This changes the effect spectrum; the native absolute EPS means the
two controls need not give the same normalized word weights.
The fourth kernel exp[-(3-sqrt(2))x^2-(3+sqrt(2))y^2] matches local kernel
curvature only, not a claimed physical Gaussian-state covariance. The six
one-mode factors multiply. Native EPS1e-6, masking, normalization, signed value,
decoder and bias are unchanged. All four share the original real12 initialization
and 81028 registered scalars; effective function capacity is not equal by counting.

## Frozen scientific contract

- New four conditions: number_coherent02, number_mixed02,
  number_mixed02_purity_normalized, curvature_matched_gaussian02.
- Ten completed controls reused, never refitted: optical fock1, parity_fock1,
  vacuum_born, gaussian3_born, rbf12, and native uniform, quantum, dephased,
  real12, softmax12. Gaussian3 is a radial slope, not a three-mode model.
- Original pinned Re-TACRED TRAIN58465/VALID19584/TEST13418 frozen features;
  seed13, 20 epochs, batch128, FP32, AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8,
  clip1, original common initial tensors and epoch order, 9140 updates/condition.
- Original per-epoch VALID all40 Macro selection, earliest exact tie. Selected
  TEST evaluated only after all new fits end. All fourteen endpoints, forty-class
  confusions, CE and thirteen descriptive repair/harm partitions retained.
- Qualification requires coherent02 TEST all40 Macro strictly greater than all
  thirteen controls. No significance or CE gate. Any failure closes this exact
  state/support/phase/embedding/control/seed/budget profile; no posthoc rescue and
  no inference that all such methods are impossible.
- Seed13 and benchmark endpoints were already exposed. A positive qualification
  cannot accept L3/T; it permits a separately frozen independent confirmation plus
  substantive same-core theoretical research. No selective additional seeds.

## Changed-code admission and execution

One scoped A6API Astra high code/math review; independent finite-number operator
and NumPy derivative checks, node/origin/large-distance, padding/schema/strict
load and actual first8 TRAIN one-update fixtures per condition. No utility
screen or repeated pause/resume/COS test matrix. No VALID/TEST input read in the
technical check. Use the existing complete-data runner, two-GPU dynamic queue,
local best/latest and ten-percent aggregate full-state COS barriers. Source,
configuration, exact registration and accepted unchanged core evidence are bound
before launch. No purchased resources or Windows feature/model/row-array download.
Retain pinned native/old-optical selected-state dependencies until paired analysis
ends; if absent restore those exact versions before analysis, never refit them.

The mathematical formulas, displacement kernels, number coherence and directional
zeros are known tools. A conditional synthetic separation is not novel theory,
universal task improvement, hardware speedup or L4-T. The gen81 gold-only frozen
query limitation motivates considering other mechanisms but does not identify
this candidate as the cause or solution. See derivation.md and prospective-study.json.
