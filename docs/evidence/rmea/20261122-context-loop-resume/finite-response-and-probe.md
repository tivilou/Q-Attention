# Protected-context: finite response and a mass-held mechanism test

Status: discovery theory and predeclared diagnostic; **L5 NOT_ACCEPTED**.
Same-source model: protected-context first-order revision633dac31 / deployed1f6e2e00.
The new model has no inherited L4 from the original Q-RMEA.

## What is already known

The model is a ridge-regularized constrained quadratic filter, closely related
to LCMV and generalized sidelobe cancellation. The already-read original
arXiv:2010.13334v1, pp7-8 Eq4-6 establishes the related inverse-covariance constrained
form. The exact original reading receipt remains in
`protected-context/original-reading-transfer.json` in owner private controls.
Frost1972 and Griffiths/Jim1982 named by consultation are leads, not newly read
full texts. Do not infer originality from quantum notation or these identities.

The previous valid96 probe already checked an entity/context mass-transfer
plus within-context redistribution decomposition. Its primary stratum contrast
was positive, but overall matched-minus-donor margin was negative. This new test
uses a preselected TRAIN cohort and DFT, and clamps each class's context mass;
it does not re-label or confirm the old valid96 result.

## Exact fixed-input statements

Let P be the retained mention-span projector, Q=I-P, C the context density,
D=C (or F C F^H for the fixed DFT control), lambda=1/8,
A=Q D Q+lambda I, delta=-A^{-1}Q D phi, w=phi+delta.

1. A is positive definite and commutes with P, so P delta=0. The vector w is
   the unique minimizer of w^H D w+lambda||w-phi||^2 subject to Pw=Pphi.
   There is no unit-norm/POVM constraint. For P=e1e1^H, phi=e1 and
   D=vv^H, v=(e1+e2)/sqrt2, w=e1-(4/5)e2, ||w||^2=41/25.
   Reinterpreting ww^H as a bounded POVM effect would require extra construction.
2. For exact retained mention vectors, raw scores including EPS remain
   unchanged; their normalized weights share a common denominator factor.
   Full protected rank8 or C=0 makes the correction zero. Truncation retains only
   the measured spectral subspace; a closing rank gap is outside the derivative
   certificate. These statements do not establish utility.
3. For D=C, first-order optimality gives
   phi^H C phi-w^H C w=delta^H C delta+2 lambda||delta||^2 >=0.
   Thus total original-context raw power falls. For DFT, only the D-weighted
   energy has this guarantee. Suppression of context is not suppression of noise.
4. With P,phi,count n fixed, replace context vector u by v. Let
   X=[v,u]/sqrt(n), J=diag(1,-1), B=QX, G=A^{-1}. Then
   Delta w=-G B (J+B^H G B)^{-1} X^H w.
   This follows by subtracting normal equations and Woodbury. It is an exact
   rank-at-most2 law for complex amplitude changes at fixed probes, not a rank
   bound for squared/normalized attention. It applies to state/covariance ports;
   raw text re-encoding may change P,other tokens and values.
5. Even Q C Q can stay fixed while the filter responds: if Qu=Qv=q,
   Delta w=-G q (v-u)^H Pphi/n. This exposes the cross-covariance term;
   it is still known constrained-filter algebra, not a new attention family.
6. For fixed per-class utilities U_i, raw scores r_i=|w^H psi_i|^2+EPS,
   Z=sum r_i and mean mu=sum r_i U_i/Z,
   Delta mu=sum Delta r_i (U_i-mu)/Z'. EPS cancels in Delta r but remains in
   Z and mu. The fixed target/competitor margin uses two separate denominators.
   Lower total power has no universal favorable margin sign.

## New discriminating experiment

Freeze the original actual native checkpoint2da52b1c. Select96 TRAIN examples
before model outputs by ID:18no_relation and2each39positive, with index tie-break.
The existing checkpoint was trained on this split: this is development mechanism
measurement, not held-out utility or independent evidence. Retain every case.

Run the same parameters/features through native, aligned C, DFT C F^H and
unprotected Q=I. For each class fix all native mention weights exactly. Normalize
only the intervention's context raw scores within context and give them exactly
the native class's context mass. Keep values/decoder/bias fixed. The competitor
is selected once by native as the largest non-target logit.

Primary population: all78positive cases. The gate is BOTH paired mean held-margin
aligned-minus-native>1e-6 and aligned-minus-DFT>1e-6. This threshold distinguishes a
nonzero mechanism direction, not a practical-gain threshold or significance test.
Report the unclamped actual aligned-minus-native margin separately and all96 as
a secondary description. No samples, controls or thresholds change after results.

A failure closes only the frozen trained-native intra-context-benefit nomination
on this fixed training cohort. It neither disproves all newly trained variants
nor justifies rescue seeds/coefficients. A pass supports this particular local
mechanism; a new trained model still needs its own utility and independent support.

Exact launch/code/pins are in `probe-review-packet.json`; Astra6.2/high both-scope
review PASS is in `probe-review.json`. One CPU scheduler diagnostic, no optimizer,
no COS and no model/data download to Windows. Runtime and measured results are
recorded separately after the authoritative task becomes terminal.
