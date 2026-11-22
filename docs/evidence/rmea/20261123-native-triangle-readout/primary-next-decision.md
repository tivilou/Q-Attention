# Primary next decision: within-word occurrence allocation, prospective only

The exact Astra6.2/high consultation is scientific advice, not acceptance.
Primary comparison with local proposal/theory/boundary/decision records found
no matching frozen fixed-margin same-word occurrence test. This does not clear
literature novelty. Fixed-margin information projection is established math;
generic projection identities alone would not qualify as L5.

The native task adapter attends to **original words**, not wordpieces. Correct
the advice's token-ID wording before implementation: use the exact original
word string and its four stored role bits as the lexical key. No lowercase,
stemming, subword merging or alternate representation. Only native valid word
positions enter; padding and transformer special positions are absent. Verify
word/offset/role alignment before execution. This prospective clarification
does not change any completed experiment or its gates.

## One hypothesis

Native attention's sentence-specific allocation between two occurrences of
the same original word contributes to correct natural decisions, beyond each
relation's usual earlier/later preference, its total weight on that repeated
word, and each occurrence's overall salience across all40relations.

For a sentence key appearing at exactly two positions, order them earlier/later.
For each relation define m_r=alpha_r1+alpha_r2 and x_r=alpha_r1/m_r. EPS makes
both weights positive. Use first8192TRAIN in stored order to estimate
q_r=(.5+sum_pairs x_r)/(1+N) for this exact lexical/role key. Use all keys with
N>=32 TRAIN pairs, not selected labels, outcomes, attention magnitude or margins.
On all19584VALID, include every sentence with at least one supported pair;
replay every supported pair simultaneously, copying all other attention weights.

Replace x_r by x'_r=sigmoid(logit(q_r)+h), with the unique scalar h solving
sum_r m_r x'_r=sum_r m_r x_r. Use float64 bisection with fixed bracket expansion
and tolerance; verify both row and column margins to1e-10 absolute. Native
logit replay and direct-vs-analytic logit change must agree to1e-8. Specifically
delta_z_r=sum_pairs m_r(x_r-x'_r)(u_r1-u_r2). Equal occurrence values make
redistribution decoder-invisible. No fitted classifier, circuit or core change.

This minimizes sum_r m_r KL(Bern(x'_r)||Bern(q_r)) under the column constraint,
by a standard one-multiplier derivative. Monotonicity gives uniqueness. It does
not imply semantic disambiguation or superiority over all positional rules;
e.g. a sentence-dependent proximity-to-subject heuristic could also pass.

## Frozen feasibility and stop

- Same native seed13 checkpoint and pinned native sources/inputs as this turn.
  First8192TRAIN/all19584VALID, no TEST access. Read only TRAIN/VALID original
  word metadata: never import TEST metadata just to obtain the two other splits.
- Need at least1000eligible VALID sentences, including250positive-gold cases.
  Gold is used only for evaluation and this declared coverage check, never key
  support, allocation or cohort selection. Also report total/positive coverage.
- On that same eligible cohort require CE(control)-CE(native)>=.005 AND
  accuracy(native)-accuracy(control)>=.003. Report both directions of correct
  decision flips, plus allVALID paired differences with ineligible differences0.
- Coverage miss or either outcome miss closes this exact hypothesis; no new
  lexical normalization, support floor, pair multiplicity, q estimator, cohort,
  threshold or seed. Numerical/identity failure is invalid/unresolved, not a
  scientific positive. No automatic retries.
- One native forward pass per split, TRAIN aggregation, one VALID replay;
  zero optimizer/checkpoint/COS writes. Code/compact reports Git; cheap private
  intermediates remain GPU caches. Use existing shared scheduler on prepared
  dual2080Ti; no fleet maintenance or lifecycle-canary campaign.

## Claim and execution status

**NOT IMPLEMENTED/NOT RUN.** Implement an isolated evaluator plugin, reuse the
existing task forward/evaluator and stable allocation wrapper, and obtain the
single scoped mechanism review plus affected margin/replay fixtures before
executing it. This is a new input-controlled alignment test, not a rescue of
the failed phase readout, relative-return fit or decision witness.

A pass would localize fixed-checkpoint contextual allocation utility beyond
the specified lexical/salience reference. It would neither explain the whole
old Q-versus-uniform gain nor inherit five-fresh-seed L4. Nearest-method
novelty, a substantive theory contribution and appropriate independent evidence
would remain open. Current L5 NOT_ACCEPTED.
