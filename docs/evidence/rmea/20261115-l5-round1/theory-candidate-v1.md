# Q-RMEA L5 research candidate: shared-effect compatibility

Status: research candidate; no L5 acceptance, no new task-utility evidence.
The accepted scoped L4 concerns head utility and local context alignment only.
This note studies the *executed* Q-RMEA pooling head at fixed preparation,
values and decoder. It is not a theorem about retraining the entire architecture.

## Exact object

For valid tokens t of sentence i, let rho_it = psi_it psi_it^†, ||psi_it||=1,
and E_r = phi_r phi_r^†, ||phi_r||=1. The frozen code uses d=8, epsilon=1e-6,
and a constrained two-layer three-qubit circuit for both pure states. Define

A_it = rho_it + epsilon I;
S_i = sum_t A_it > 0;
H_ir = sum_t u_irt A_it, with u_irt = decoder_r^T value(h_it).

Then alpha_irt = phi_r^† A_it phi_r / (phi_r^† S_i phi_r), and
z_ir - bias_r = phi_r^† H_ir phi_r / (phi_r^† S_i phi_r).
Only valid tokens enter these sums. These are forty separate binary effects;
sum_r E_r need not be I. There is no shared forty-outcome POVM in this code.

## Known reductions, not novelty claims

The trace form |phi^†psi|^2 = Tr(E rho) is a known quantum-kernel identity.
With F_it = S_i^(-1/2) A_it S_i^(-1/2), sum_t F_it=I. Defining
w_ir = S_i^(1/2) phi_r / sqrt(phi_r^† S_i phi_r) gives
alpha_irt = w_ir^† F_it w_ir. This is a sentence-specific token POVM
representation, not a relation POVM. The transformed w depends on the sentence.
The circuit-constrained phi does not range over every unit vector.

For any fixed sentence/relation, the unconstrained maximum/minimum of z-bias
are the largest/smallest eigenvalues of S^(-1/2) H S^(-1/2). These are ordinary
generalized Rayleigh-quotient facts, not a new quantum advantage theorem.
Likewise a proposed support-set mass uses H=sum_{t in support} A_it and the
same quotient. Support is not automatically semantic evidence.

## Candidate result: compatibility of a shared effect across sentences

Fix a relation r and a finite nonempty set of sentences. Let
lambda_i = lambda_max(S_i^(-1/2) H_ir S_i^(-1/2)),
G_i = lambda_i S_i - H_ir (positive semidefinite), and
L_i = lambda_max(S_i). For weights a_i>0 with sum_i a_i=1, set

F = lambda_min(sum_i a_i G_i/L_i).

For every *common normalized* phi (and hence every circuit-reachable phi),

sum_i a_i [lambda_i - (phi^†H_ir phi)/(phi^†S_i phi)] >= F >= 0.

Proof: each gap equals phi^†G_i phi/(phi^†S_i phi). Its numerator is
nonnegative and its denominator is at most L_i. Sum the resulting quadratic
lower bounds and apply the ordinary minimum-eigenvalue bound.

Moreover, all per-sentence maxima can be reached by one unconstrained phi iff
intersection_i ker(G_i) contains a nonzero vector. For the actual head, that
intersection must also contain a circuit-reachable ray. Proof: zero gap iff
G_i^(1/2)phi=0. Strictly positive weights require every gap to vanish. In finite
dimension, F>0 iff that common kernel is trivial. This certificate can be loose
and small positive numerical values require an error bound or a declared tolerance.

Scope: this is a fixed-feature, fixed-value/decoder, one-relation compatibility
statement. H_i can have either sign. It gives a logit or chosen-support bound,
not a CE/Macro improvement bound. Independent per-sentence optima may be
unattainable under the circuit; they are a relaxation. Retraining preparation,
decoder or values changes the problem. The same shared-query difficulty can
occur in classical pooling; no exclusive quantum benefit follows.

## Exact counterexample and limits

In d=2, two valid orthogonal token states |0>,|1> give S=(1+2epsilon)I.
Sentence 1 utilities are (1,0), sentence 2 utilities (0,1). The individual
maximum is lambda=(1+epsilon)/(1+2epsilon). For every shared phi, the two scores
sum to 1, so their mean is 1/2. The gap to the two independent maxima is exactly
1/[2(1+2epsilon)] = F. Independent queries reach each maximum. If sentence 2
instead has utilities (1,0), F=0 and the same |0> query attains both maxima.

Embedding into d=8 retains the exact opposing-utility mean=1/2. The displayed
L_i-based spectral certificate becomes looser because epsilon supplies the six
orthogonal directions; do not transfer the d=2 tightness to the actual circuit.
Identical prepared states with differing utilities can arise from a lossy
preparation map, but this example does not establish that it occurs in Re-TACRED.

An independent change of query per sentence is a stronger different model.
It cannot inherit the old L4, and its extra capacity must be controlled before
any task-utility claim. This note does not recommend a formal experiment yet.

## Discriminating question, not a confirmed prediction

Does *shared-effect compatibility*, beyond per-sentence relaxation and generic
attention concentration, explain which fixed-head development cases have a
large achievable allocation/logit gap? An analytical distinction exists between
aligned and opposing fixtures above. Whether it matters on real text is untested.

First audit nearest work on generalized Rayleigh quotients, sums of ratios,
attention expressivity and measurement compatibility. This result uses standard
linear algebra; its novelty and substantive value remain unresolved and may fail.
Then use a frozen development-only sample selection to compare the certificate
with actual selected-head gaps and matched classical controls. Labels/outcomes
must not choose favorable groups. Keep its role descriptive; an outcome-guided
utility claim needs a separately frozen validation. Do not rerun a stopped
complement, contrast or spectrum family under this new name.
