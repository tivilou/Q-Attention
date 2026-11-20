# Gauge-covariant token-graph transport: one speculative mechanism

Status: analytical candidate for cheap source qualification; no implementation, numerical execution, task gain, novelty clearance or L5 acceptance. Independently suggested by gpt-6.1-sol at medium, then analyzed by the primary. The fixed time tau=1/4 is the consultant's initial rule; it has not been tuned on results.

## Construction and baseline limit

For N valid normalized token states psi_t in C^8, let

    G_ts=psi_t^dagger psi_s,
    K_tt=0, K_ts=G_ts*abs(G_ts)^2 (t!=s),
    H=K/||K||op if K!=0, otherwise H=0,
    U(tau)=exp(-i*tau*H).

K is Hermitian. For query phi_c, set a_ct=psi_t^dagger phi_c,
D_c=sum_t abs(a_ct)^2+N*epsilon, epsilon=1e-6, and

    rho_c=(a_c a_c^dagger+epsilon I_N)/D_c,
    alpha_c(tau)=diag(U rho_c U^dagger),
    z_c(tau)=sum_t alpha_ct(tau) u_ct+bias_c.

This preserves nonnegative normalized attention and the original utility readout. At tau0, it is exactly the original EPS-smoothed attention. The epsilon identity part is invariant under U. No learned residual or new fitted coefficient is introduced. It does add a token-index register and denser computation. The cubic Gram kernel is a proposed design, not a source theorem or optimized choice.

The matched dephased control is

    alpha_inc,c(tau)=abs(U(tau))^2 alpha_c(0).

The squared absolute value is elementwise. It is a doubly stochastic transition map applied to original attention, retaining the same graph, tau and utilities. Equivalently it first removes off-diagonal entries of rho_c and then evolves it. No arbitrary phase convention is changed for this control.

## Why this is not R85's fixed phase-sensitive erasure

Let token representatives be rephased by psi_t -> exp(i theta_t) psi_t and let Q=diag(exp(i theta_t)). Then

    G' = Q^dagger G Q, K'=Q^dagger K Q,
    a'_c=Q^dagger a_c, H'=Q^dagger H Q,
    U'=Q^dagger U Q, rho'_c=Q^dagger rho_c Q.

Operator norm is unchanged, including the zero case. Therefore both coherent and dephased output attentions are exactly unchanged. A query-global phase cancels as well. Unlike R85's fixed erasure projector, the transport operator transforms with the token convention. The final map is a function of token/query projectors, not arbitrary independent vector phases. This does not supply a free quantum implementation: controlled preparation, Hamiltonian construction and readout resources would need a new explicit contract.

## A task-decision-linked current identity

Holding representations and utilities fixed, differentiation at tau0 gives

    dz_c/dtau = (2/D_c) sum_t u_ct Im[a_ct^* sum_s H_ts a_cs]
              = (2/D_c) sum_(t<s) (u_ct-u_cs) Im[a_ct^* H_ts a_cs].

Pair currents are antisymmetric; their sum preserves normalization. For nonzero K, each imaginary factor is proportional to

    abs(<psi_t|psi_s>)^2
      * Im[<phi_c|psi_t><psi_t|psi_s><psi_s|phi_c>].

This is a gauge-invariant three-state Bargmann product. Triple-product and quantum-current identities are known tools; they are not claimed as new L5 theory. The identity relates the new path directly to decoder utilities and a correct-class versus fixed-rival margin. Its sign can help or harm. If all states can be represented as real in a common basis up to individual phases, the first derivative is zero; finite-time changes may still occur at higher order.

## Exact finite-step two-token source witness

Choose an effective two-dimensional subspace. Set psi_0=|0>, psi_1=(|0>+|1>)/sqrt2 and phi=(|0>+i|1>)/sqrt2. Native squared overlaps are both1/2. Since G_01=1/sqrt2, the stated normalized graph is H=sigma_x. Up to an irrelevant query-global phase,

    a=(1/sqrt2,(1+i)/2), D=1+2epsilon,
    alpha_coh,0(tau)=1/2+sin(2tau)/(2sqrt2 D),
    alpha_coh,1(tau)=1/2-sin(2tau)/(2sqrt2 D).

Native attention and the dephased control are exactly(1/2,1/2), because their input distribution is uniform and abs(U)^2 is doubly stochastic.

Give class1 utilities(1,-1), bias0; null utilities0 and bias1/4; all other utilities0 and biases-2. Native and dephased predict null. At the fixed tau1/4, coherent class1 logit is sin(1/2)/(sqrt2 D)>1/4, so it predicts1. A strict elementary bound suffices: sin(1/2)>23/48 and sqrt2<3/2 imply the logit>23/[72(1+2e-6)]>1/4. This is a finite decision difference, not merely a favorable infinitesimal own-score response.

The witness is realizable by the original depth2/three-qubit circuit up to one shared final CNOT-ring unitary W. Set all first-layer angles0. Token0 has all second-layer angles0; token1 has second-layer wire0 RY=pi/2, RZ=0. Query has second-layer wire0 RY=pi/2 and RZ=pi/2. All other angles0. Outputs are W times the stated token states and the stated query up to its global phase exp(-i*pi/4). All nonzero raw angles are atanh(1/2), finite under pi*tanh(raw). Two linearly independent feature rows allow the prepare layer to produce the two token settings. A single value coordinate and decoder row implement(1,-1), with other decoder rows zero. This is an analytic parameter construction, not a numerical circuit test or actual checkpoint observation.

Changing the example's true label to null makes the same transport harmful. The witness establishes distinguishability from the native and matched incoherent map, not classwise accuracy, Macro improvement, source-trained reachability or phase necessity on text.

## Cheapest next test and stop boundaries

First qualify exact source behavior, not a new training loop: tau0/EPS/padding/N1/K0, state rephasing, token permutation, common-H graph control, the finite source witness, and current finite differences. Numerical computation must use shared scheduling after exact mechanism/lifecycle review. For a fixed-checkpoint forward-only screen there is no need to differentiate the operator norm. Joint training would separately require a frozen policy for eigenvalue/zero degeneracies and gradient tests.

If qualified, freeze one existing native checkpoint and one fixed development cohort, and compare native, coherent tau1/4 and dephased tau1/4 with unchanged values/biases. Report fixed positive39 Macro, per-class confusion, positive-to-null/null-to-positive/positive-to-positive counts, and correct-versus-preintervention strongest-rival margin changes. All units and labels used for selecting direction remain discovery, not independent confirmation. No optional tau/sign/seed/epoch rescue. A normal weak result is inconclusive for all trainable variants but stops automatic progression of this exact deployment hypothesis.

Coherent benefit must be distinguished from ordinary graph contextualization; dephased parity weakens that attribution without erasing ordinary utility. Known complex-kernel/quantum-walk-attention collisions remain unresolved. No formal full-data handoff until eligibility and exact code review pass; full formal experiments remain collaborator-owned.

## Original source boundary

Farhi and Gutmann, Quantum Computation and Decision Trees, quant-ph/9706062v2, PDFp5 SectionII Eq2.1 and p6 Eq2.8/2.12/2.13: distinguish classical Markov evolution exp(-Ht) from unitary quantum amplitudes exp(-iHt) and normalized Born transition probabilities. Their graph is a real tree Laplacian, unlike the proposed complex cubic-Gram graph. Their penetrability/search statements are not imported as semantic or computational advantage here. The mechanism is exactly classically simulable. Broader attention-neighbor search remains incomplete; bounded metadata searches establish neither priority nor absence of collisions.
