# Quantum structure transfer: controlled order and process incompatibility

Status: exploratory mathematical triage, not a selected experiment or L5 contribution.
Owner clarified that the 100-question request concerned a different project; it is withdrawn here. Re-TACRED remains unchanged.

## Target problem and evidence

Relation extraction requires assigning evidence to the correct ordered arguments. The actual encoder already supplies continuous subject/object distances and span flags. Native attention is forty independently normalized binary Born scores, followed by a signed value decoder. Context-dephasing damages the fixed original model, but neither that intervention nor synthetic circuit separation establishes a missing capability of trained classical models. The previous natural-binding40 screen has insufficient label contrast (upper bound7<required8); do not resample or use its three operational witnesses as prevalence or mechanism evidence.

The transfer question is whether composition of operations can provide a useful, role-conditioned token ordering beyond state overlap. It is not whether a familiar quantum formula can be attached to attention. Native, classical biaffine/path encoders, and exact classical circuit simulation are separate comparators. Classical simulability does not exclude a useful inductive bias; it excludes an unqualified computational separation claim.

## Original sources actually read

1. Araujo, Costa and Brukner, arXiv:1401.8127v5, main text PDFpp1-4; p2 equations rendered. Under a promised proportionality between permutations of black-box unitaries, coherent routing exposes an order label. The source distinguishes query cost from running time; its exponential classical-time statement is a conjecture. Appendices unread. https://arxiv.org/abs/1401.8127
2. Zeng and Coecke, EPTCS221(2016),pp67-75, official arXiv copy1608.01406; PDFpp1-8, including p7 Eq5/Theorem5.2 and diagrams rendered. Compositional tensor contractions and direct similarity readout avoid some intermediate materialization under explicit sparse-vector/oracle/QRAM, precision and tree assumptions. Exact arXiv revision unresolved; p9 unread. https://arxiv.org/abs/1608.01406
3. Sajjan et al., Physical Review Research5,013146(2023), DOI10.1103/PhysRevResearch.5.013146, publisher PDFpp1-3. Eq5 defines the known OTOC; this paper studies imaginary components and correlation in an RBM learner for physical target states. Later proofs/experiments not read. It does not establish a text-attention advantage. https://doi.org/10.1103/PhysRevResearch.5.013146

Crossref query results are discovery metadata, not a comprehensive novelty search. One query returned429; web search transport returned404. No absence-of-prior-art conclusion follows.

## Transfer obstruction 1: the source promise can disappear on normalization

For word state psi_t, relation effect phi, and two entity unitaries A,B, write

    p_theta(t) = |<phi|(AB + exp(i theta)BA)|psi_t>|^2 / 4.

Under AB=exp(i chi)BA, set q_t=|<phi|BA|psi_t>|^2 and

    c = |exp(i chi)+exp(i theta)|^2/4.
    p_theta(t) = c q_t.

For c>0 and EPS=0, normalization cancels c exactly. With native EPS>0, S=sum_t q_t>0 and T valid tokens,

    alpha_t(c) = lambda(c) q_t/S + (1-lambda(c))/T,
    lambda(c) = c S/(c S+T EPS).

S=0 or c=0 gives uniform attention. Thus the promised order label changes only uniform shrinkage in this factorized construction. It does not supply a new token-specific pattern beyond q. This is conditional algebra, not a statement comparing arbitrary different A/B while holding q fixed.

For fixed scalar utilities u_t, the logit difference between c and1 is

    [lambda(c)-lambda(1)] * [mean_q(u)-mean_uniform(u)].

Its magnitude is bounded by osc(u)*|lambda(c)-lambda(1)|. A mathematical signal in a circuit is insufficient unless the actual attention and decoder retain it.

Without the proportionality promise, K=(AB+exp(i theta)BA)/2 yields p=|<K^dagger phi|psi_t>|^2. This is a subnormalized rank-one effect with ||K||<=1. It can be a richer entity-conditioned query, but is not a separation from general complex queries. A controlled-gate implementation also does not inherit a black-box indefinite-order query advantage.

## Transfer obstruction 2: squared commutators lose orientation

For token and relation unitaries U,V and a fixed input |0>, the known process incompatibility is

    p_minus(U,V) = ||(UV-VU)|0>||^2/4
                 = (1-Re C(U,V))/2,
    C(U,V) = <0|V^dagger U^dagger VU|0>.

It lies in[0,1], but p_minus(U,V)=p_minus(V,U). Therefore this particular score cannot itself encode the sign of reversing the two operations. This is not a proof that a role-conditioned model using it is symmetric under raw subject/object swaps: the contextual encoder and role features can break that symmetry.

An oriented quadrature p_quad=(1+Im C)/2 instead obeys p_quad(V,U)=1-p_quad(U,V). This is a known interference measurement, not a new theorem or established mechanism of semantic argument direction. It requires a task-linked learning hypothesis and classical controls. Merely using its imaginary part does not establish L5.

## Source-specific state-access boundary

Native U(a)=F L2(a) F L1(a), with F the directed CNOT ring. Let C0=U(0)=F^2 and center U_tilde=C0^dagger U. Common centering leaves native state fidelities unchanged and gives identity at zero angles.

Take token operations I and RZ(pi/2) on wire0, and relation operation RY(pi/2), with all other angles0. Both tokens prepare the same density and have native fidelity1/2. Their squared-commutator scores are0 and1/4. All nonzero angles use finite raw tanh parameter atanh(1/2). The token-dependent difference survives native EPS normalization.

This is only a fixed state-port versus full process-access witness, already conceptually present in the closed relative-return work. It is not a new expressivity separation from retrained classical encoders, proof of natural-text reachability, utility, novelty or quantum hardware advantage. Do not relabel the closed partial-swap, ordered-entity, relative-return or scattering families as successful through this identity.

## Decision before GPU admission

Close importing the proportional-order promise as a new allocation mechanism, and close squared-commutator magnitude as an intrinsic direction sign. Preserve these cheap transfer constraints. The oriented process score remains an unselected hypothesis. An exact mechanism implementation and review would be needed before any pilot; no R2 code or new training is admitted by this card.

The next selected task should identify a natural role-dependent operation, what information its composition preserves, and a strong classical binding comparator. A standard OTOC formula or synthetic completion witness alone does not settle that selection. No quantum speedup, L5 nomination, checkpoint/COS write, or full experiment is claimed here.
