# Entity-terminal routing: exact reductions and a coherent-scattering lead

Status: theory-only exploratory. No method execution, hyperparameter selection, task benefit, novelty clearance or L5 acceptance. Old scopedL4 remains unchanged. Independent Astra builder proposed the DC-current composition; the primary derives the limitations below and the alternative coherent-resolvent construction AFTER receiving that proposal. Do not label the latter an independent consultant invention.

## Verified carryover and changed causal link

Current main86828ab9 closes frozen operator-variance diagnostic. GammaQ<0 and all-controls positive-direction gateFAIL; no positive-filter/scalar rescue. Existing fullseed13 linear coherent head remains a one-seed positive lead, not an inheritednewmethodL4. R84 readout incompatibility is a different oldcohort/descriptive observation. Current mathematical question changes the path from relation-specific representation/value to class score through entity-source/sink propagation. Class competition remains one40waysoftmax; no label is available at inference.

## A. Consultant DC-current proposal and exact limitations

Let q_i>0 be the already normalized Born weight for one query, sumq=1; K_ij=|<psi_i,psi_j>|^2, c_ij=q_i q_j K_ij for i!=j. For distinct singleton entity terminals s,o, clamp v_s=1,v_o=0 and solve sum_j c_ij(v_i-v_j)=0 internally. Let m_i=sum_j c_ij |v_i-v_j|, w_i=m_i/sum_k m_k. Zero-current disconnected components require an explicit component policy; no such implementation is approved here.

1. A branch attached through only one articulation and containing neither terminal has constant potential and zero m. This is a standard maximum-principle consequence. It proves invariance to utility changes IN THAT FIXED READOUT only, not to raw text changes: frozen RoBERTa contextual h, preparation, q and the prior linear transport can carry effects from that token elsewhere.
2. On any positively weighted tree, only the unique s-o path of L edges carries current I>0; endpoints have m=I and its L-1 interior nodes m=2I. Thus w_s=w_o=1/(2L), each interior w=1/L, all others0. Every c and hence every q cancels. A gain on such graphs cannot identify a relation-specific Born-conductance mechanism.
3. On a complete graph with constant K_ij=kappa>0, let a=q_s+q_o and b=1-a. Every interior potential equals q_s/a. Endpoint m=kappa*qs*qo/a, interior m_i=2kappa*q_i*qs*qo/a. Hence w_s=w_o=1/[2(2-a)] and w_i=q_i/(2-a) internally. This is exact entity upweighting plus rescaled original context weights, not a new path computation. It is the mandatory simple null if a future flow study is attempted.
4. Swapping the terminal clamps gives v'=1-v and exactly the same m/w at fixed q/K. The pooling rule alone is direction-blind. Role-conditioned features/values may still distinguish relation direction; do not claim reversal equivariance or direction novelty.
5. Newman cond-mat/0309045v1 pp5-7 already supplies current-flow/random-walk betweenness, Laplacian solve and cul-de-sac suppression. His endpoint convention counts source/sink current separately; the consultant's uniform incident-current convention assigns half endpoint weight relative to that convention. This difference and fixed weighted pair versus all-pair average are explicit, not a claim of a new conservation law.
6. Zhang/Qi/Manning1809.10185v1 p4 explicitly motivates preserving off-path negation/conjunction. Our learned dense graph is not their syntactic tree, so this is a transfer warning, not direct empirical rejection. It defeats the unsupported assumption that a pendant token is semantically irrelevant.

Conclusion: DC-current composition is an exploratory engineering option, NOT an independent L5 contribution on these known facts. Do not launch a frozen-head null and then generalize it to the trainable family.

## B. Primary coherent-scattering proposal (not yet selected for implementation)

Use the same circuit states psi_i in C^8 and query Born weights q_ri. Let Psi have columns psi_i, G=Psi^H Psi, D_r=diag(sqrt(q_r)). For lambda>0 and real onsite values e_ri, define

 H_r=lambda D_r (G-I)D_r + diag(e_r).

A possible future onsite map is a bounded map of existing relation-token utilities; its precise scaling/map is UNFROZEN, not available for outcome-driven selection. For now e is a real algebraic input. With fixed energy E, eta>0 and terminal lead strengths gamma_s,gamma_o>0, define

 B=eta I+(gamma_s P_s+gamma_o P_o)/2,
 R_r=[E I-H_r+i B]^-1,
 T_r=gamma_s gamma_o |(R_r)_{so}|^2,
 ell_r=log(T_r+epsilon)+b_r, epsilon>0.

This is standard single-channel passive Green-function/Landauer transmission used as a possible new task readout, not a new transport law. No target labels enter H/terminals/R/T at inference. Real tasks with multisubword entities need a prospectively frozen anchor rule; no experimental contract exists yet. A general multichannel extension uses trace(Gamma_s R Gamma_o R^H) and need not have T<=1. No no-relation, task-utility or independent-confirmation claim follows.

### B1. Exact low-rank reduction and conditioning

Let X=sqrt(lambda) Psi D_r (8xT), F=diag(E-e_ri+lambda q_ri+i B_ii). Then

 E I-H_r+iB=F-X^H X,
 R=F^-1+F^-1 X^H [I_8-X F^-1 X^H]^-1 X F^-1.

Because Im(F-X^HX)=B>=eta I, the full inverse exists and ||R||_2<=1/eta. F also invertible, so the8x8 Schur factor exists. This is Woodbury, a known tool, not quantum speedup or independent novelty. It makes a token-level coherent resolvent computationally plausible without doing40 T-cubed inversions perbatch. Actual numerical parity/performance remains untested.

### B2. Exact side-branch distinction from DC

Consider a real4-site star ordered s,b,o,d with nonzero edges a:s-b, b0:b-o, c:b-d. Onsite energies e_s,e_b,e_o,e_d. For A=E I-H+iB define z_i=E-e_i+iB_ii. By eliminating d, the b onsite denominator becomes z_b-|c|^2/z_d. The cross-terminal cofactor gives

 R_so = a*b0*z_d / det(A)

for this real symmetric convention. Its transmission can change when only e_d changes, although d is a DC pendant. For eta=0, positive terminal leads, and E=e_d, z_d=0 but det(A)=-|c|^2 z_s z_o !=0; hence R_so=T=0. This is an exact anti-resonance, not automatically an asymmetric Fano lineshape. With c=0 the dangling degree is decoupled and must be removed before taking eta=0; direct backbone transmission at that E is nonzero generically. With eta>0 the exact zero is generally lifted; an eta0 algebraic result is not a practical zero at finite damping.

The intervention changes only e_d at a fixed chosen Hamiltonian. A language interpretation additionally requires the prepared states to realize suitable couplings and an actual token perturbation to affect these onsite values; the star establishes neither fact for the trained checkpoint or natural text. This is established molecular side-group physics, see1203.5269v4 p2 Eq2 and cond-mat/0609109v1 p3 Eq1. The proposal transfers that known causal link: a side token can modulate entity-to-entity evidence by interference rather than be dropped by DC pooling. No evidence yet says that learned language graphs map useful modifiers to these spectral roles.

### B3. Response, gauge and competition boundaries

For real onsite e_i, dR/de_i=R P_i R. A stable exact logit derivative is

 d ell_r/de_ri = 2 gamma_s gamma_o Re(conj(R_so)*R_si*R_io)/(T+epsilon).

It can have either sign, unlike the failed positive operator-variance response. The resulting CE derivative remains sum_r(p_r-1[r=y]) d ell_r; signed response by itself guarantees no task improvement. If onsite values also change q/Psi, the total derivative includes those paths; do not confuse this partial with full training gradient.

For token gauge Psi->Psi U, U=diag(exp(i chi_i)), require the actual Gram input a=Psi^H phi to transform to U^H a, and the spectral/operator-norm normalizer to be unitarily invariant. Then exp(-i tau (G-I)/scale)a transforms to U^H times itself, so its normalized Born q is unchanged. Also require the onsite map to depend only on gauge-invariant real values, not arbitrary token phases. Under those assumptions H->U^H H U, diagonal B commutes, R->U^H R U, and T invariant. Without those upstream assumptions only the fixed-q/e resolvent invariance is proved. On a tree, all hopping phases are removable by such a gauge: the star anti-resonance does not certify phase resource or quantum advantage. Classical coherent-wave and rational-kernel models can reproduce it. Nonzero cycle-phase effects need a separately frozen comparison; do not assume them from this toy.


### B4. A parameter-sign prediction for ordered entity readout

The adversarial consultant proposed the following three-site check; the primary has independently expanded its two cofactors below. This is an exploratory capacity result, not novelty acceptance or executed evidence.

Let three valid tokens be ordered s,o,d. Keep upstream q, onsite e and all Hamiltonian entries fixed while exchanging ONLY the two readout leads, with equal strengths gamma>0. Uniform damping eta>0 stays fixed. This is a controlled readout intervention; swapping raw entity roles in a real sentence need not preserve those upstream quantities. Write a=h_so, b=h_sd, c=h_do, z_d=E-e_d+i eta. Lead equality means the full A=EI-H+iB is the same in both readouts. The cofactors are

 R_so=(a z_d+b c)/det A,
 R_os=(conj(a) z_d+conj(b c))/det A.

Put k=a*conj(b c)=h_so h_od h_ds. Expanding the two modulus squares gives

 T_so-T_os = -4 gamma^2 eta Im(k)/|det A|^2.

The same diagonal/normalization and the same softmax bias must be used when interpreting a score difference. With epsilon>0 the sign of log(T_so+epsilon)-log(T_os+epsilon) equals the sign above. A real cycle, a tree (all phases removable), or eta=0 wherever the inverse exists gives equality. For eta>0 a nonreal three-cycle gives a strict directional difference. This is a lossy/nonreciprocal coherent-wave capacity; do not call it time-reversal breaking by labels or quantum-exclusive advantage.

An architecture-level prepared example exists. After a common final CNOT ring W, choose
 psi_s=W|000>, psi_o=W[(|000>+|100>)/sqrt2], psi_d=W[(|000>+i|100>)/sqrt2].
The circuit can prepare these up to independent global phases: all first-layer angles zero; in the second layer only first-wire RY/RZ vary as (0,0),(pi/2,0),(pi/2,pi/2). Its pi*tanh parameter map reaches pi/2 at finite rawatanh(1/2). Common W preserves overlaps. Thus G_so G_od G_ds=(1+i)/4, and

 Im(h_so h_od h_ds)=lambda^3 q_s q_o q_d/4>0.

The eps-positive normalized Born rule after the SAME linear transport guarantees q_s q_o q_d>0 for any fixed valid query. Ignoring role-feature columns while constructing this synthetic preparation permits a change of readout anchors without changing q, but does not assert that the actual trained head ignores roles. A feature-to-angle map admitting those prescribed rows is a circuit-family construction, not an actual trained checkpoint or natural-text witness. The readout must expose its anchor inputs separately from the frozen role-conditioned representation for this controlled test. Therefore T_so<T_os for this chosen orientation, at ANY positive eta/gamma/lambda, with real onsite energies. Conjugating the prepared states changes the sign of the cycle imaginary part; upstream q may change under the fixed-sign linear transport, so predict the opposite DIRECTIONAL SIGN only, not equal-magnitude paired transmissions.

This strict prediction distinguishes the proposed final complex-hopping readout from the same fixed-q/e electrical incident-current null and a real |G| hopping readout, both direction-blind under the controlled equal-lead reversal. It does not exclude a general classical directed or coherent-wave network, and phase information can still affect q upstream in both arms. The loop product is a standard Bargmann invariant; the cofactor identity is a standard resolvent calculation. Their task-specific composition remains unestablished for novelty and utility. Next assess nearest chiral/magnetic graph-learning works before implementation.

## Before-code success/kill and independent empirical boundary

First obtain adversarial verification of the reductions, zero/damping assumptions and gauge convention; compare precise original-paper equations. A correct derivation is L0/tools, not L5. Stop an independent novelty nomination if the actual task mechanism is only a renamed known Green-function model with no new task-relevant discriminating result. Any eventual implementation needs a new exact-source both review, bounded numerical/gradient checks, and a frozen full-data matched-control design with localbest/latest and COS10%recovery. Use one threeGPU scheduling group, no optional seeds/checkpoint rescue. Repeating inference on the already reused validation/test is not an independent replication. A future fixed-cohort confirmation must explicitly declare independent training units and its conditional assumptions; it cannot be described as fresh-cohort generalization. Current owner authorization permits preparing this work, not claiming that it has run or achieved utility.
