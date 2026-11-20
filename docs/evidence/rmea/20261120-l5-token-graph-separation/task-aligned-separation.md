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

## Executed correspondence check

The selected 2080 Ti scheduler task f07e278bd76f456c99206cf099ea39bc, attempt 7d7d625d09764068a3a4732f37646a81, completed with exit zero and a quiescent receipt. Ten existing/additive unit tests passed in 4.72624 seconds at registered revision 5af89f6350ca122b796cae11e37dc6148e98e7c7. No model, protocol, training budget or formal gate changed.

The additive test uses the reviewed full-rank nine-key witness, middle-token utility for one synthetic class, constant rival logits and a midpoint decision threshold. It calls the implemented routing functions and compares the candidate's CE to exact generalized-eigenvalue bounds. Cubic CE was 1.6306877321, versus relaxed native minimum 1.6333255673 and relaxed dephased minimum 1.6394148548. The smaller separation was 0.00263783515. Explicit extreme-eigenvector queries attained the comparator optimum logits within 5.551115e-17, checking the bound's orientation and attainability. The chosen synthetic utility/target makes this a construction, not an observation of naturally learned useful routing.

The three-key dark-token probability was 0.000956456752 for both coherent and initially dephased routes. Replacing the last key by an isolated orthogonal state reduced it to the smoothing floor 5.714275918e-7, as predicted. See task-aligned-numerical-check.json. This checks the exact null rather than attributing the activation to coherence.

## Measurement-theory comparison: candidate deductions

The following deductions passed focused mathematical review by gpt-6.2-astra at medium effort (resp_0f79fad270701190016b00afe4fe3887d190a532127270a39a). See measurement-comparison-review.json for the corrected source metadata and original BLOCK. They are tools and limitations, not new L5 acceptance criteria.

**Exact measurement representation, including smoothing.** Set E_i=psi_i psi_i†+epsilon I and D=sum_i E_i. Native attention is a POVM F_i=D^(-1/2)E_i D^(-1/2) on the reparameterized query state sigma=D^(1/2)rho D^(1/2)/tr(D rho). This transformation is bijective on density matrices because D>0. It is shared by all fixed-key routes.

Define an isometry V from C^d to C^N direct-sum (C^N tensor C^d): its first block is Psi†D^(-1/2), and its second block consists of N identical row blocks sqrt(epsilon)D^(-1/2). Then V†V=I. For outcome i, measure the projector onto signal coordinate i together with noise block i. Applying W=U direct-sum I_(Nd) before these projectors gives exactly the implemented coherent attention, retaining epsilon. Dephasing the outcome-index sectors before W gives the initially dephased route; its probabilities equal T times native probabilities, with T_ij=|U_ij|^2. Thus it is a classical stochastic post-processing of native measurement at fixed keys.

This is a Naimark-dilation representation, not a new dilation theorem or a claim that the current simulator implements a hardware measurement. The auxiliary dimension is only a convenient explicit construction, not an optimized resource bound.

**A generic informativeness measure is not an advantage certificate here.** Skrzypczyk and Linden, arXiv:1809.02570v1, p2 Eq7, define robustness R(F)=sum_i ||F_i||op-1. On p3 they show every rank-one POVM reaches the dimension-dependent maximum d-1, and Eq13 relates this quantity to an optimally chosen discrimination game against guessing without measurement. The benchmark is not our trained native head or its NLP task.

If the key frame M=Psi Psi† is full rank and epsilon=0, both native and coherent effects have rank at most one and sum to I. Hence both have R=d-1; in our d=8 construction both equal7. A change in probabilities, or an advantage on one state pair, need not improve this generic robustness.

With epsilon>0, write either effect as f_i f_i†+epsilon D^(-1). Since the effects sum to I, their traces sum to d. Also ||f_i f_i†+epsilon D^(-1)||op >= ||f_i||^2. Therefore

    d-1-N epsilon tr(D^(-1)) <= R <= d-1.

This applies to both routes with the same D. Their robustness difference is at most N epsilon tr(D^(-1)); for a frame lower bound M>=m I it is at most Nd epsilon/(m+N epsilon). This upper bound can be loose and does not prove which route is larger. It is not useful when the bound is large. The rank-one equality requires an invertible M at epsilon=0; no singular-limit interchange is asserted.

**Specific information can differ despite that equal generic maximum.** Use the same full-rank nine-key construction (e0, v, e1, e2,...,e7), v=sqrt(3)/2 e0+1/2 e1, and two physical queries phi_plus=(e0+i e1)/sqrt(2), phi_minus=(e0-i e1)/sqrt(2). Every original key is real, so these queries have identical native probabilities; applying T preserves that equality. The released generator uses the Hermitian operator norm sqrt(7)/4; its normalized path edges are r,s with r^2+s^2=1, so the rotation angle is tau. The middle coherent signal numerator is w w† (the smoothed numerator is w w†+epsilon I, and the POVM effect is its conjugation by D^(-1/2)), with w=cos(tau)v+i sin(tau)x and x=r e0+s e1 as above. Direct expansion gives

    p_coherent(middle | phi_plus) - p_coherent(middle | phi_minus)
      = -sqrt(3/7) sin(tau)cos(tau)/(3/2+9 epsilon),

which is nonzero at tau=1/4. Consequently no fixed classical post-processing of the native outcome statistics reproduces the coherent measurement on both inputs. Even a nonlinear function of the identical exact native probability vectors cannot distinguish this pair. With equal priors the native measurement's optimal guessing probability is1/2, while coherent measurement permits1/2+||p_plus-p_minus||_1/4>1/2.

This elementary constructed pair does not establish a new general resource theorem, superiority for every ensemble, or a task benefit from real inputs. Its varying query states are abstract measurement inputs; in the NLP head, class queries are learned global parameters. It therefore does not exhibit two NLP examples whose task information is lost by the implemented native head. It also does not rule out processing raw quantum states or raw features, changing the keys/queries/values, or applying an additional physical preprocessing unitary. The state-pair/post-processing comparator is different from the earlier arbitrary-query probability-region comparator; neither should be silently substituted for the other.

The substantive research question is whether the learned graph preserves information needed by the fixed task that the comparator's pooling loses. Generic coherence, maximal rank-one robustness and an optimally chosen constructed game are insufficient evidence. The pending formal utility and frozen swaps must answer their declared task question before further promotion.

## Finite grids for the whole local pooling family

Focused mathematical/additive-test review PASS from gpt-6.2-astra at medium effort, response resp_0bba9c85ee830734016b00b316592087d1b24a1b4542f13aed; numerical execution is pending. See local-pooling-family-review.json. This changes neither model nor frozen collaborator protocol.

The implemented native class logit has the form z(x_1,...,x_N)=sum_i n_i(x_i)/sum_i d_i(x_i), with d_i>0. The class bias is absorbed by replacing each local numerator f_i u_i by f_i(u_i+b). Queries, preparation and local linear values may take any fixed parameters. Partition the tokens into two groups A,B. On a crossed input grid (A_i,B_j), every such logit must satisfy

    Z_ij = (a_i+b_j)/(c_i+d_j), with c_i+d_j>0.

This remains a necessary condition even when each group's numerator and denominator are arbitrary joint functions inside that group. It therefore covers every retrained native head at the same frozen-feature interface, rather than fixing its original keys, query or values. It does not cover cross-group contextual feature recomputation, global input-conditioned queries/values, another outer nonlinear decoder, or coherent value aggregation with unrestricted global value states.

For a four-by-four grid, use the numerator gauge a_0=0 and denominator gauge c_0=0, subtracting those constants from all a,c and adding them to b,d. Let t contain a_1..a_3,b_0..b_3,c_1..c_3,d_0..d_3. Let I be the sixteen-by-seven row/column incidence matrix after those omissions. Define

    A(Z) = [-I, diag(vec Z) I].

The sixteen equations A(Z)t=0 encode rational-additive representation exactly; a physically valid representation has a nonzero t because its cell denominators are positive. Consequently full column rank14 excludes every such representation. A column-rank test is only a numerical witness at finite precision, not a proof of rank for the exact real-number implementation.

A useful conservative approximation certificate also follows. Let k have seven numerator zeros and seven denominator entries1/4, so k^T t is the mean of the sixteen cell denominators. Normalize it to one. If A has full column rank, its thin SVD A=U S V^T gives the exact real-arithmetic minimum

    r_min = min_{k^T t=1} ||A t||_2
          = 1 / ||S^(-1) V^T k||_2 > 0.

For any rational-additive comparator Zhat with positive cell denominators, normalize those denominators to mean one. Its parameters obey A(Zhat)t=0 and

    A(Z)t = (vec Z - vec Zhat) elementwise-multiplied by vec D.

Positive D has sum16, hence ||D||_2<=16. Therefore

    max_ij |Z_ij-Zhat_ij| >= r_min/16.

This is a lower bound for approximating these sixteen pre-softmax logits, not a classification-error, loss or generalization guarantee. Numerical evaluation is conditional on the represented logit matrix and linear-algebra error; no exact certified arithmetic is claimed. The native route supplies an explicit positive-denominator null as an internal validity check.

The fixed test uses seed20261121, one untrained actual head, four synthetic three-token blocks per side, their sixteen crossed six-token inputs, class0 only, float64/complex128 and the unchanged four routes. No label, optimization or search is involved. Its declared structural criterion is native relative minimum singular value below1e-10 and cubic above1e-8; a failed or inconclusive grid is retained without searching seeds or changing tolerances. Other graph routes are reported without a required direction: breaking local pooling separability would not by itself identify cubic nonlinearity or coherence as necessary. Reuse the existing GPU unit recipe once after focused mathematical review.

## Next use

First analyze the pending formal utility gate and its existing frozen swaps/readout bounds. Only if the result and available approved traces justify it, evaluate loss-aligned bounds as an explicitly separate post hoc diagnostic. Do not rerun the eight-example canary, alter the formal gate, or request another training budget to obtain this evidence.
