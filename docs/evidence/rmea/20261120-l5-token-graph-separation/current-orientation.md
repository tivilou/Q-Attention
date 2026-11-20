Focused mathematical review: PASS, gpt-6.2-astra medium, response resp_03d448239b7b47f6016b00c240a5c887d190a434451376e904. See current-orientation-review.json. No numerical/empirical execution is claimed.

# Born-invisible orientation and task-weighted coherent current

Prospective analysis of the unchanged token-transport head at development f0fb563370090b0fcf19331b78119efd9f2daefb. This is a mechanism probe, not a new architecture, empirical finding or L5 acceptance. The formal source, four controls, tau=1/4, data, budgets, sample selection and gates remain frozen.

## Exact interface and parameter involution

On valid tokens let A have rows psi_i^dagger, G=AA^dagger, a_r=A phi_r, D_r=||a_r||^2+N epsilon, epsilon=1e-6. For q=1 or3 define K_q,ij=G_ij|G_ij|^(q-1) off diagonal and zero diagonal, H_q=K_q/||K_q||op, U_q(t)=exp(-itH_q). The multi-token graph must be nonzero; a singleton is the identity case. Let V_r=diag(u_ri), where utilities and bias are those of the unchanged value/decoder head. Then

 z_coh,r(t)=b_r+[a_r^dagger U_q(t)^dagger V_r U_q(t) a_r+epsilon tr(V_r)]/D_r,
 z_dep,r(t)=b_r+tr[V_r U_q(t)(diag(|a_r|^2)+epsilon I)U_q(t)^dagger]/D_r.

The actual circuit applies RY then RZ on each wire and real CNOT permutations, for two layers. Define C on the actual finite head parameters: negate output rows1,3,...,11 of prepare.weight and prepare.bias (the RZ coordinates); negate effect[...,1]; preserve RY coordinates, value, decoder and bias. Because tanh is odd, C conjugates every prepared key and class-query state, on every feature input, without changing the architecture or parameter count. C is an involution. Consequently:

* Native Born probabilities, normalizers, logits and CE are exactly invariant under C for every input and label.
* G->G*, a->a*, H_q->H_q*, U_q,C(t)=U_q(-t)*.
* Both coherent and initially-dephased outputs at C(theta),t equal their original outputs at theta,-t. The linear coherent route has the same relation.

This is a native parameter-identifiability symmetry and an exact in-family intervention. It is NOT separation between two NLP inputs for all native parameters, a learned-family expressivity theorem, or proof that coherent routing is useful. The dephased route is generally NOT even in t on a complex graph.

## The first-order distinction is query-conditioned current

Define the real, antisymmetric pair current

 J_r,ij=2 Im[H_q,ij a_rj a_ri*]/D_r = -J_r,ji.

Direct differentiation of da/dt=-iH_q a gives dp_ri/dt=sum_j J_r,ij, and

 j_q,r := dz_coh,r/dt at0
         = i a_r^dagger[H_q,V_r]a_r/D_r
         = sum_(i<j)(u_ri-u_rj)J_r,ij.                    (1)

The initially-dephased derivative at0 is exactly0, since its initial density and V_r are diagonal; this is only a first-order statement. Its odd response can start at order t^3 on complex cycles. Constant utilities, a zero current, or a circulation with zero divergence give zero first-order logit response even when phases or coherence exist. Current magnitude alone is not alignment with useful values.

For the actual Gram generator,

 H_q,ij a_rj a_ri* = |G_ij|^(q-1)/||K_q||op
                   * <psi_i|psi_j><psi_j|phi_r><phi_r|psi_i>.       (2)

The last product is the three-vertex Bargmann invariant. It is unchanged by independent phase choices of all keys and the query; it conjugates under C. Thus j changes sign while every native probability stays fixed. Cubic and linear routes have the SAME triangle phases. Their current difference is exactly

 j_3,r-j_1,r = (2/D_r) sum_(i<j)(u_ri-u_rj)
   Im[<psi_i|psi_j><psi_j|phi_r><phi_r|psi_i>]
   * (|G_ij|^2/||K_3||op - 1/||K_1||op).                (3)

Consequently cubic transport reweights already present signed currents; neither larger affinity powers nor a nonzero phase alone imply a better task direction. This does not revive the stopped entity-fusion partial-swap/commutator family: the current generator acts on token amplitudes and is the already frozen candidate.

## Finite-time odd-response certificate

Let sigma_r=(a_r a_r^dagger-diag(|a_r|^2))/D_r, Delta_r=max_i u_ri-min_i u_ri, and d_r(t)=z_coh,r(t)-z_dep,r(t). Then d_r(0)=0 and d_r'(0)=j_q,r. Since ||H_q||op<=1,

 |[d_r(t)-d_r(-t)]/2 - t j_q,r|
       <= (2/3) Delta_r ||sigma_r||_1 |t|^3.             (4)

Here ||.||_1 is the matrix trace norm, not entrywise L1. Proof: differentiate tr[sigma U^dagger V U] three times; subtract the midpoint of the utility range from V. Three commutators have operator norm at most8||V-midpoint I||op=4Delta. Trace/operator norm duality bounds the third derivative by4Delta||sigma||_1. Taylor remainders at t and -t give (4). For actual t=1/4 the right side is Delta||sigma||_1/96. If |j_q,r|>Delta||sigma||_1/24, the sign of this coherent-minus-dephased odd logit contrast is certified by the current. Otherwise the bound is inconclusive, not a negative utility result. No tau search is proposed.

## Connection to task loss and prospective use

At the same frozen head's native logits z(0), p=softmax(z(0)), and a post-generation label y,

 d CE_coh(t,y)/dt at0 = sum_r (p_r-1[r=y]) j_q,r.        (5)

The initially-dephased loss derivative at0 is0. A negative value in (5) is a local descent direction only; it does not certify improvement at1/4, accuracy, generalization or necessary mediation of a trained-model gain. C flips (5) exactly. If a distribution of native fitted heads is C-invariant, the expected signed odd response is0; this is conditional on that distributional symmetry, not an observed property of our training. It says nothing about the mean even response.

The pending formal report has four trained heads and a frozen24-example trace sample. If the returned approved trace fields permit exact reconstruction, analyze ALL those samples without selecting a favorable class/subset: native/coherent/dephased/linear logits on each fixed head, conjugate/time-reversed counterparts, currents, Eq4 residual/bound, and Eq5 versus actual fixed-tau CE differences. Report signed disagreement and dispersion, including failures. Labels are analysis-only after logits exist. Do not use them to choose a query, orientation, tau, checkpoint, head, token mask or sample. These are secondary prospective mechanism diagnostics, not changes to the formal success gate. Technical-canary traces alone cannot establish trained task alignment. Missing fields require a separately reviewed export-only handoff; no collaborator-host access and no retraining for this analysis.

## Literature scope and claim ceiling

Rabei et al., arXiv:quant-ph/9907080v1, p6 Eqs2.8-2.9 and p7: cyclic inner-product products and their ray-phase invariance are established geometry. Zimboras et al., arXiv:1208.4049v3, pp2,4-5 Eq2 and Methods: complex hopping, conjugation/time-reversal, gauge transformations and loop phases are established transport tools; the inspected study initializes localized site states. Its site-state phase-removal claims do not automatically apply to our coherently prepared query amplitudes unless the initial state transforms too.

Our current-specific mapping, actual finite-circuit parameter involution and loss-weighted prediction connect those tools to this implementation. They do not constitute novelty clearance. Eq1/4/5 are standard differentiation/norm tools, not standalone L5 theorems. A substantive same-core contribution still needs independent full-data task utility and evidence that the proposed orientation/current mechanism explains it beyond the matched controls. No empirical result, code execution or new rating is claimed here.
