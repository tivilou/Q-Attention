# Mention-protected context filtering: bounded frozen-head probe

Retain the trained native Q-RMEA quantum state preparation, class queries, utilities and bias. For each example let S be the entity-token state span, Pi the projector onto its orthogonal complement, R the empirical non-entity covariance, and lambda=1/8. Solve

min_w w^dagger R w + lambda ||w-q||^2, subject to w-q in S-perp.

With delta=w-q, the unique solution is delta=-(Pi R Pi+lambda I)^-1 Pi Rq. Pi commutes with Pi R Pi+lambda I, so delta is in S-perp. Multiplying the stationarity equation by delta^dagger yields

q^dagger Rq-w^dagger Rw = delta^dagger R delta + 2lambda||delta||^2 >=0.

Thus all protected complex amplitudes are preserved and non-mention total raw affinity decreases in the exact span model. The source epsilon floor is kept. Mention mass consequently increases when context power decreases and mention mass is positive. However, non-mention tokens are not known noise and may carry relation triggers; none of this implies a better logit or margin.

Writing m0,m1 for mention mass, U_M for its unchanged conditional utility, and U_C0,U_C1 for context conditional utilities, the exact logit decomposition is

Delta logit = (m1-m0)(U_M-U_C0)+(1-m1)(U_C1-U_C0).

With truncated numerical span the omitted term is m1*(U_M1-U_M0); the implementation bounds original mention amplitude residuals, audits discarded singular values, and explicitly checks the logit identity tolerance. The solve is certified only for frozen inference in this stage. No optimization derivatives, training utility or novel theorem are asserted.

This is a transfer of known constrained filtering, not a claim to invent LCMV or null-space projection. The scoped Guo et al. original is arxiv2010.13334v1 pp7-8 Eq4-6; distributed communication assumptions are not transferred. Frost1972 is DOI-metadata-only in the current evidence. A substantive new quantum contribution remains open.

The contracted diagnostic uses96 cases from already exposed official validation:18 no_relation and2 per other label selected by id_hash before outcomes. A label-blind donor covariance uses the same recipient mention span and readout. The fixed native strongest non-target competitor defines all margins. The primary descriptive contrast is the mentions-higher minus context-higher difference of matched-minus-donor margin changes, null when a stratum is empty. Its sign is a falsifiable prediction, not a guarantee. No significance or L1-L5 acceptance follows from this slice.

One CPU scheduler task, zero GPUs, zero optimizer steps, one attempt. Source/input/checkpoint pins, protected responses, context power, utility decomposition and saved-state replay are checked. Capture is limited to two preselected private diagnostic fixtures. This is not sample-trace.v2, a completed Case Study, a full-data fit or formal-ready implementation. Future matched-budget training, if justified, must separately compare native/matched/donor, retain minimal local/COS recovery, and use the three-card grouped runner; it inherits no old scoped L4.
