# Revised question: task direction versus perturbation curvature

Status: exploratory question, not an accepted theoretical contribution.
This changes the question after the first shared-effect certificate failed to
explain the accepted alignment endpoint. It does not change the model or L4 card.

For any logits z, perturbed logits z', label y and p=softmax(z), q=softmax(z'),

CE_y(z') - CE_y(z) = (p-e_y)^T (z'-z) + KL(p || q).

This is the known Bregman identity of log-sum-exp. It is not a new theorem.
The first term measures the change along the current task-loss gradient; the
second is nonnegative and does not depend on the observed label. Therefore
positive CE damage from a permutation, by itself, does not identify beneficial
first-order label alignment. It remains evidence of dependence under that
intervention, the narrow interpretation already accepted for Q-RMEA.

For Q-RMEA the exact logit change is
delta_r = sum_t (alpha'_rt-alpha_rt) u_rt.
The original fresh5 reports give aggregate CE differences but do not identify
the two terms. No allocation certificate derived from independent per-sentence
optima can supply those missing observations.

For a random label-blind permutation, define zbar=E[z_perm]. Then

E[CE_y(z_perm)]-CE_y(z_actual)
 = CE_y(zbar)-CE_y(z_actual)
   + E[LSE(z_perm)]-LSE(zbar).

The last term is a nonnegative Jensen penalty, independent of y. This is a
different estimand from the original single frozen permutation and must not
replace its gate. The first-order term and the mean-prediction benefit are
also distinct; neither should be silently substituted for the other.

## Analytical falsification fixture

Three valid nonentity tokens, alpha=(.4,.2,.4), utilities=(-1,0,1), logits=(0,0),
give zero actual target logit. The fixed cyclic permutation (1,2,0) gives target
logit +.2. On two otherwise identical cases with labels 0 and 1, mean loss grows
by log(cosh(.1)) > 0 while the average directional term is exactly zero and
accuracy remains .5. Averaging all six permutations yields zero mean logit but
strictly positive CE inflation. The construction uses no outcome-based
permutation choice. Smoothed Born weights can realize these positive token
probabilities as an unconstrained two-dimensional state construction; actual
Re-TACRED behavior and circuit reachability are not established by this fixture.

## Next research action

Read original attention-faithfulness and identifiability work before calling
this an innovation. Determine whether a useful, nontrivial identification result
can be obtained for the particular relation-measurement mechanism. Current
identities and the witness are diagnostics, not sufficient for L5.

If a distinct research claim survives, freeze a development-only, outcome-blind
cohort and complete comparison inventory before a bounded fixed-head replay.
Report both signed direction and KL, failure cases and exact state/source/data
identities. No new full training or cache restoration is justified merely by this
question. A later claim about independent task benefit needs appropriate new
evidence and collaborator-only formal execution; the old reported CE statistic
does not prospectively confirm a newly derived decomposition claim.
