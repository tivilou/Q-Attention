# Connecting fixed-key separation to classification loss

Status: focused mathematical review PASS from gpt-6.2-astra at medium effort, response resp_0a517162eb0d94b5016b00a84eaab487d28395355cac7645f7; no required corrections. See task-aligned-theory-review.json. These are analysis tools for the existing cubic candidate, not a new architecture or an L5 novelty claim. The collaborator protocol remains frozen.

## What the existing evidence does and does not establish

The six fixed one-update technical-canary traces gave 138/240 simultaneous native/dephased full-vector support witnesses. Separately, the same producer's eight hash-ordered test examples gave zero native readout upper/lower exceedances among 320 class readouts, at tolerance 1e-8. The two test traces are the first two examples of that eight-example set; this is not a comparison of aggregate rates on identical cohorts. No new inference or training was performed to read these saved results.

The original summary hash is d6c9137c662de9d298577ca1102c6be48a4169d1dae00b64f818fd8290a648b4. The producer is the already accepted one-update parent canary, not a trained efficacy study. All its frozen-route accuracies were zero on eight examples. This cannot reject the pending full-data candidate.

A probability vector outside the native region need not produce a logit outside the native readout range. The learned value/decoder map can discard the separating directions. Thus geometry alone does not establish a useful decision advantage.

## Exact logit box for unrestricted physical queries

Fix one example's keys, real utilities u[r,i] and class biases b[r]. Let

    E_i = psi_i psi_i† + epsilon I,
    D = sum_i E_i > 0,
    A_r = sum_i u[r,i] E_i,
    z_r(phi_r) = b[r] + phi_r† A_r phi_r / (phi_r† D phi_r).

Allow each class r its own arbitrary unit physical query phi_r. This is a relaxation of our finite circuit family and does not impose a single shared choice over multiple examples. Define

    L_r = b[r] + lambda_min(D^(-1/2) A_r D^(-1/2)),
    U_r = b[r] + lambda_max(D^(-1/2) A_r D^(-1/2)).

Then the exact attainable logit set in this relaxation is the Cartesian product of [L_r,U_r]. Generalized Rayleigh quotients bound every coordinate. Conversely, every point between the two extreme eigenvalues is obtained by a normalized superposition of extreme eigenvectors in whitened coordinates, mapped back by D^(-1/2) and renormalized. The class queries are independent, so these choices can be made simultaneously. Pure queries already fill each interval; mixtures do not enlarge it.

Consequently, if a coherent logit vector lies inside this box, some independently chosen native physical queries reproduce its entire logit vector on this example, even when its attention vectors lie outside the native probability region. This does not imply that the implemented native circuit or one globally trained query set realizes those choices. Zero numerical exceedances at a finite tolerance do not prove exact box membership.

The same construction applies to the initially dephased comparator, using

    E_i^dep = sum_j |U_ij|^2 E_j.

Their sum is still D. The transport U is fixed by this example's keys and is independent of the physical class query.

## Exact best achievable CE in that relaxation

For a fixed post-generation target y, cross-entropy decreases with z_y and increases with every rival logit. Therefore

    CE_native_min(y) = log(1 + sum_{r != y} exp(L_r - U_y)).

This minimum is attained by maximizing the true-class quotient and independently minimizing every rival quotient. Evaluate it with log-sum-exp. Likewise the maximum is log(1 + sum_{r != y} exp(U_r - L_y)). A strictly correct top-1 decision is achievable exactly when U_y > max_{r != y} L_r; ties require the actual tie policy.

If the fixed cubic output's CE is strictly below both the native and dephased minima, no choice of their physical class queries can match that loss while these keys, utilities and biases remain fixed. This connects a separation to an actual classification objective. It is a sufficient diagnostic, not a required gate for ordinary trained-model utility, a causal-mediation proof, or an advantage over retrained keys/values. A null result remains compatible with useful optimization or inductive-bias effects.

Targets are used only to analyze already generated logits and bounds. They must not choose the inference query, route, checkpoint, sample, hyperparameter or training budget. These prospective quantities are not added retrospectively to the frozen formal success gate.

## A graph-context distinction with an explicit coherence null

QCSAM's inspected Eq6/Eq28 uses a normalized complex value sum sum_j <K_j|Q> |V_j>. With a fixed query, adding or changing a key orthogonal to it contributes zero to this sum when all other keys/values and the query are held fixed. This is only an interface-level statement; globally changed value states or upstream contextual representations invalidate it.

For the selected cubic route use keys (e,a e+b f,f), query e, with a=sqrt(3)/2,b=1/2 and orthonormal e,f. Write r=3sqrt(3)/(2sqrt(7)), s=1/(2sqrt(7)), c=cos(tau). The initially zero-score last token has transported amplitude

    a_last = (c-1) s r - i sin(tau) s a.

Its probability exceeds epsilon/(1+a^2+3epsilon) by

    s^2 ((1-c)^2 r^2 + sin(tau)^2 a^2) / (1+a^2+3epsilon).

Replacing the last key by g orthogonal to e and f makes that token isolated and removes this excess. However, in this real three-key example, the initially dephased route has exactly the same last-token probability: the two amplitude contributions have a zero real cross term. Indirect activation of a low-score token therefore does not itself identify a benefit from coherence. This is a discriminating null, not a new-method superiority result.

## Next use

First analyze the pending formal utility gate and its existing frozen swaps/readout bounds. Only if the result and available approved traces justify it, evaluate loss-aligned bounds as an explicitly separate post hoc diagnostic. Do not rerun the eight-example canary, alter the formal gate, or request another training budget to obtain this evidence.
