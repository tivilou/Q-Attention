# Geometric context and selective dephasing in the original head

Motivation: original fresh-five development VALID has classwise heterogeneous
benefit/damage against matched controls. That does NOT identify numeric cues,
distance, distraction or phase as the cause. Do not select favorable classes.

New hypothesis: coherence in context strictly between the input subject/object
spans and coherence in other nonentity context have different task effects.
This is not a duplicate-word/four-role occurrence test, a shared-pooling interval
certificate, two-group gradient balancing, or a new filtering/readout architecture.

Define the two masks ONLY from input entity spans in the source half-open
[start,end) convention: bridge is valid nonentity
positions strictly between the disjoint spans; outer is all other valid nonentity
positions. Both masks exclude subject/object tokens. Overlapping spans are
ineligible. A row with either region empty is ineligible and all interventions
are identity for that row; no evaluation rows are dropped. No distance threshold,
word list, model score, label or validation outcome enters mask construction.

First qualify all first8192 existing TRAIN word/span mappings against cache
roles/offsets. Proceed only if eligible rows>=2048, eligible positive rows>=1000
and >=10 positive classes each have>=16 eligible rows. Counts use labels only
after masks are fixed; do not rescue a failed qualification by changing regions,
cohort or thresholds. Qualification has no model calls, VALID/TEST, or fitting.

If support passes, consider exactly four frozen-head inference conditions:
native; dephase bridge; dephase outer; dephase all context. Only masked token
density operators change rho -> diag(rho). Entity states, populations, values,
effects, parameters and EPS remain fixed. No mixing coefficient or training.
The comparison asks about this fixed source model's use of coherence at the
readout interface, not reachable text edits or retrained dephased alternatives.

Prospective utility hypothesis for one bounded full development VALID pass:
outer dephasing must improve positive39Macro and CE versus native, and improve
both versus bridge and all-context dephasing, tolerance1e-8. Use ALL19584 rows,
with the frozen identity rule above. All conditions reported; no region/metric/
step/seed/checkpoint rescue. A miss closes this exact frozen intervention, not
all locality or channel-aware learning. Even pass is not independent evidence,
quantum necessity, substantive novelty or L5. Review changed scientific code and
exact launch contract before any model computation; reuse accepted core/runtime.

Source identity for a mask M: write q_rt=p_rt+EPS and d_rt=M_t*(pdiag_rt-p_rt).
At dephasing strength lambda, Nlambda=N0+lambda*sum(d)>0 and the exact logit
change is lambda*sum(d*(u-pooled0))/Nlambda. Thus raw coherence loss alone does
not determine task response; the value-centered interference alignment does.
This is elementary normalized-weight algebra, a TOOL not a novel theorem.
Whether a broader source-specific theory of context-dependent utility can
be substantive remains open and requires nearest-work comparison.
