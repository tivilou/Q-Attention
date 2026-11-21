# Three-token determinantal pooling: exploratory theory screen

Status: exploratory_only; Astra6.2/high PASS_WITH_SCOPE for mechanism algebra and exact fixture; fixture executed exit0; no task-utility, priority, quantum-exclusivity or L5 claim. No model implementation or training is authorized by this note. Original negative scattering is closed and untouched.

## Scope and prior art
Independent Astra6.2/high proposed full three-subset volume weighting and a positive pair-only control. Primary full-text reading now confirms ordinary quality/diversity DPP attention in Li et al., arXiv:1909.10852v2, pp3-4, equations7-10 and sections4.1-4.2; k-DPP subset law is Kulesza and Taskar arXiv:1207.6083v1, p57 equation175. Our law below is exactly a complex Hermitian 3-DPP with L=diag(sqrt(q)) K diag(sqrt(q)). Neither that law nor determinant-attention is a new contribution. Identity/version checking is separately recorded; PDF theory title page contains an internal May18,2022 date alongside arXivv1 stamp, so do not silently equate internal date with submission date.

## Definitions
For n>=3 unit free states psi_i in C^d, G_ij=<psi_i,psi_j>, K=(I+G)/2, positive class-specific qualities q_i, and bounded learned utilities |u_i|<=U, define triples T, w_T=prod_{i in T}q_i, v_T=sum_{i in T}u_i/3.
B_T=1-(|G_ij|^2+|G_jk|^2+|G_ki|^2)/4; C_T=Re(G_ij G_jk G_ki)/4.
det(K_T)=B_T+C_T. Full law p_F(T) proportional w_T(B_T+C_T); pair control p_B(T) proportional w_T B_T; readout z=b+E[v_T]. Positive B>=1/4; actual detK>=1/2 follows eigenvalues G>=0, traceG=3 and product(1+lambda)>=1+sum lambda=4.
The actual retained linear_coherent source has transport weights, not just native Born weights: a_i=<psi_i,phi>, coherent alpha=(|a exp(-i tau H)^T|^2+EPS)/(sum|a|^2+n EPS); H is offdiag Gram divided by its Hermitian operator norm, tau=.25. Bounds below allow arbitrary positive q and apply to either weight choice. The definition does not resolve which choice a future model would use.

## Proposition1: phase contribution bound uniform in length and qualities
Let rho=max_{i!=j}|G_ij|<=1, dplus=rho^3/(4-3rho^2), dminus=dplus if rho<=1/2 else 1/26.
For every triple, -dminus <= C_T/B_T <= dplus. Consequently with A=1-dminus, B=1+dplus,
TV(p_F,p_B) <= (sqrt(B)-sqrt(A))/(sqrt(B)+sqrt(A));
|z_F-z_B| <= 2U times this TV bound.
These are upper bounds; no claim of jointly attainable readout sharpness. At small rho the score bound is U*rho^3/4+O(rho^5).
Proof: write a,b,c for edge magnitudes, s=a^2+b^2+c^2, r=abc. Positive ratio <=rho^3/(4-3rho^2). For a negative cycle, PSD gives s+2|Re cycle|<=1. Thus magnitude <=min((s/3)^(3/2),(1-s)/2)/(4-s). First branch increases, second decreases; intersection s=3/4 gives 1/26. For rho<=1/2, s<=3rho^2 gives smaller bound dplus. Let R=1+C/B in[A,B], p_F=p_B R/E_B R. The standard extremal bounded-likelihood-ratio inequality gives stated TV bound; bounded observable range2U gives readout bound. This uses known total-variation reasoning and elementary PSD constraints, not claimed a new universal theorem.

## Proposition2: synthetic cubic-order witness, including retained linear weights
Take n=4, 0<rho<1/2. G_3 is Hermitian circulant with clockwise edges rho exp(i theta/3), and G_4=diag(G_3,1). It is PSD by min eigenvalue>=1-2rho. Query cross-overlap vector is c times all-ones, c^2=(1-2rho)/8. For joint5x5 Gram [[G4,c1],[c1^*,1]], the inverse quadratic form c^2 1*G4^(-1)1 is <=c^2*4/(1-2rho)=1/2; hence its Schur complement is >=1/2 and the matrix is positive definite. Thus states+query exist in <=5 complex dimensions, embeddable in C8; no actual fixed ansatz/text reachability claim.
Native q_i=1/4. All-ones vector on circulant block is an eigenvector, so original offdiag-Gram spectral-normalized coherent transport changes only a common phase on the first3 overlaps and leaves the fourth unchanged. Coherent q_i is also1/4, with the same EPS smoothing. Utilities u4=1, others0; they are free bounded scores, not a proven shared-head reachable fixture.
Let d=1-rho^2/4 and t=1-3rho^2/4+rho^3 cos(theta)/4. Then z_F-b=d/(t+3d), while z_B uses t without cycle term. theta=0 versus pi leaves all edge magnitudes, both q choices and utilities fixed, changes full score by
d*(rho^3/2)/((4-3rho^2/2)^2-rho^6/16)=Theta(rho^3).
This tests algebraic phase sensitivity of free input states, not task benefit or complex necessity. Conjugation preserves Re cycle, so the method cannot distinguish globally conjugate configurations. Real signed Gram matrices can also have different cycle signs.

## Proposition3: expressivity tradeoff independent of training
alpha_i=P_F(i in T)/3<=1/3; sum alpha=1. This follows from the readout's equal average over each selected triple. For n=3 every triple law has a single subset: z=b+(u1+u2+u3)/3, independent of q and G. A useful single token cannot receive original q_i near1, so preserving utility aggregation is not the same as preserving useful routing capacity. This is an analytic limitation, not measured inferiority on Re-TACRED. For n<3 consultant fallback uses all tokens, also uniform. Adding an ad-hoc blend to avoid this changes the mechanism and is not authorized by this note.
K contains an identity regularizer; its Gram representation augments psi_i with a token-specific orthogonal direction e_i. It is not an operator measurement solely on the original3qubits. No hardware advantage/exclusivity claim.

## Decision boundary
Before any GPU candidate, verify exact formulas/witness and show a substantive contribution beyond known DPP attention. Current default: do not promote the unmodified 3-DPP as L5 direction. Utility is unresolved, not disproved. Qualification/common-protocol baseline comparison references remain unlocated, so this remains exploratory_only regardless of toy success. Retain unchanged scopedL4 and completed negative full-seed13 evidence.

## Executed outcome and primary disposition
Unchanged fixture SHA256 b5306d36fa4a1dac63cc604367696268fa22b7c6d611f5442cf79ddd9af7752e passed160 randomized Gram/quality/utility cases, ten PSD query witnesses, coherent/native uniform-quality checks, determinant versus pair decomposition, enumerated versus polynomial 3-DPP marginals and concentration-cap tests. Largest observed random-case TV was0.014613905291241056; this is a synthetic descriptive value, not a task result. The emitted -1/26 endpoint was literal; its analytical equality follows from the all-minus-one-half triple, but no independent numerical endpoint-equality assertion was executed. Scope is exactly local_synthetic_fixture_only.

Primary disposition: DEPRIORITIZE_UNMODIFIED_3DPP_AS_L5_LEAD. Keep it as an exploratory comparator or conditional phase-sensitivity tool. The known DPP construction and concentration cap do not prove poor task performance, but leave no demonstrated substantive new core or reason yet to spend a full3GPU experiment. No theory-to-utility or oldL4 inheritance claim. Next bounded question: can unchanged original pooling receive a substantive shared-deletion margin theory without confusing internal token suppression with re-encoding raw text, and is that distinguishable from existing attention erasure/robustness certificates? This is unselected, with nearest-work and realizability audits still open.

Git representation note: the public fixture has its final CRLF normalized to LF; see public-line-endings.json for both SHA256 values and exact-byte/AST equivalence. Executed result/review retain the private original source hash; no semantic change or new review was introduced.
