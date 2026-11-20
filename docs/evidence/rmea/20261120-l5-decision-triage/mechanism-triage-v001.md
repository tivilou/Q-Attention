# Decision-linked source triage v001

Status: primary exploratory analysis; no scientific numerical execution, selected implementation, novelty clearance or L5 acceptance. Four source-level routes are compared; this is not four tested ideas. The class-frame question was already noted in R99 and is developed here without relabelling its history.

## Audited starting point

The frozen cross-fold screen improved CE and accuracy but failed its positive39 Macro gate. Against full, null-correct increased by21 and positive-correct decreased by12. The class-centered screen had a different failed joint-gate pattern. These observations identify an endpoint mismatch, not a proven common cause. Repeatedly explored development data and six absent positive classes prevent an independent confirmation claim. No closed screen is replayed or rescued.

## Comparison and cheapest decisions

| Route | Causal change | Cheapest useful check | Primary disposition |
|---|---|---|---|
| Class-frame coupled POVM | Competitor queries jointly change each class's token routing, with original utilities/readout retained | Exact degeneracies, classical equivalence and a competitor-query intervention; then source-specific conditioning/gradient review | Retain one speculative design lead. Known PGM alone supplies no new L5 theorem or task evidence |
| Input-label joint density posterior | Replace attention/readout with kernel-weighted label-density conditioning | Derive directly from source Eq4/7/8 and compare with known kernel classifier | Known-method control; not selected as a new core contribution |
| Semantic null as unambiguous inconclusive outcome | Impose zero error for every conclusive class outcome | Compare40 pure states with dimension8 and separate semantic null from abstention | Reject this exact proposal under its stated assumptions, not all noisy POVMs |
| Metric-aware class-cost rule | Change class decision costs using confusion geometry | Check fit-only cost identifiability and leakage; compare the same frozen rule across heads | Known decision-control question, not a new quantum mechanism |

## One retained speculative lead: class-frame competition

For40 normalized query pure projectors P_c, dimension d8 and the unchanged epsilon=1e-6, define

    A_c=P_c+epsilon I, H=sum_c A_c,
    E_c=H^(-1/2) A_c H^(-1/2),
    q_ct=Tr(E_c rho_t), alpha_ct=q_ct/sum_valid_t q_ct,
    z_c=sum_valid_t alpha_ct u_ct+bias_c.

H is positive definite since H>=40epsilon I; E_c are positive and sum_c E_c=I. No new learned parameter, residual or post-measurement epsilon is proposed. This is the known square-root measurement of the uniform regularized class ensemble. R1's known PGM acted on the token frame; changing the axis gives a different routing dependency but no automatic novelty.

For pure token states, the exact classical expression is

    q_ct=abs(phi_c^H H^(-1/2) psi_t)^2
          +epsilon*norm(H^(-1/2) psi_t)^2.

This explicit counterpart constrains attribution, not ordinary utility eligibility. A competitor-query intervention can change alpha_ct even if phi_c, tokens and utilities remain fixed; native routing has no such dependency. This is an interface prediction, not a semantic or post-training improvement claim.

Cheap analytic limits:

- If H=cI, the common factor cancels and alpha is exactly native.
- With one valid token, all class attentions equal1.
- If all queries coincide, H=40A and every E_c=I/40. Token attention becomes uniform. Native shared-query attention need not be uniform.

The last limit permits both help and harm. With two orthogonal token states and common query e0, native attention is ((1+epsilon)/(1+2epsilon),epsilon/(1+2epsilon)); the coupled attention is(1/2,1/2). Set true-class1 utilities(1,-1), all other utilities0, null bias1/2 and other biases0. Native predicts1; coupled predicts null. Set true-class1 utilities(-1,1), null bias-1/2 and all other class biases-2; native predicts null while coupled predicts1. These are abstract interface witnesses, not executed source-circuit fixtures or evidence of learned task benefit.

Therefore a PGM discrimination guarantee or POVM completeness cannot justify a universal CE/Macro theorem. The task hypothesis would be that reducing redundant query evidence selectively improves positive-class decision margins. It requires evidence of that selectivity and matched controls; query decorrelation by itself is insufficient.

Before implementation: establish actual source representability and a stable inverse-square-root derivative at repeated eigenvalues. Naive eigendecomposition autograd can be ill-conditioned even though the matrix function is smooth. The identical-query and tight-frame limits are required gradient fixtures. Any solver is fixed before results; no post-hoc regularization tuning. This is a design question, not a launch-ready candidate. The current decision is to compare independent suggestions first, not spend another full implementation cycle on this lead automatically.

## Two source transfers with immediate boundaries

Gonzalez et al., arXiv2004.01227v2, PDF title Classification with Quantum Measurements; landing title Supervised Learning with Quantum Measurements. From the joint empirical density in Eq4, input projector Eq6, normalized conditioning Eq7 and partial trace Eq8, the output is a label-state mixture weighted by abs(input overlaps)^2 and normalized by their sum. Proposition2 explicitly identifies the kernel interpretation. This is a known classifier family. The p4 M normalization notation differs between Eq15 and Eq17; the normalized formula here is derived from Eq4/7/8 instead of copying those constants. No claim of a new classifier or task gain follows.

Chefles, quant-ph/9807022v1, SectionII pp3-5: for pure input states, zero-error discrimination with positive conclusive success for every state requires linear independence. Forty pure class states in dimension8 violate that condition. More than d noisy POVM outcomes remain legal; the theorem does not forbid them. A dataset's semantic no-relation label is also not a physical failure-to-identify outcome. The exact null=inconclusive proposal is rejected; a separately posed approximate decision problem would need its own design.

Metric-aware costs: Narasimhan et al., ICML2015, Consistent Multiclass Algorithms for Complex Performance Measures, original pp1-4. Their confusion-geometry methods already address nondecomposable metrics. Optimal confusion/costs are not given by a repeatedly explored development confusion matrix. A separately frozen fit-only control could clarify a bottleneck, but it is not a new L5 contribution on its own.

## Iteration accounting

Primary comparison:4 routes,1 retained speculative lead,0 implementations,0 new scientific runs. Source/proof screening can close a precisely stated proposal; an ordinary short-training null remains inconclusive on the whole mechanism. Proofs, publication and reviews are not new mechanism trials. Next decision must incorporate independent discovery and source/collision checks before any exact code review. No shared-skill or scheduler change, full formal experiment, checkpoint/COS action or promotion is implied.

## Independent pass and primary selection

An independent problem-only medium pass supplied one additional family: gauge-covariant coherent token-graph transport. The primary derives an exact finite source-circuit two-token decision witness; a separate six-check medium math review has zero must-fix items with explicit omitted-source qualifications. A new GPT6.2Astra medium assignment resolves common unitary/linear query absorption and exact classical contextual re-encoding. See companion records.

Design priority1 is token-graph transport, because the baseline limit, coherence-removal control and finite decision difference are explicit. Priority2 is the older class-frame lead, retained without automatic implementation. QMC/metric costs remain known controls and the stated zero-error null interpretation fails its premise. This is four primary source routes plus one independently suggested family, zero numerical trials. No candidate is yet execution-eligible or novelty-cleared.
