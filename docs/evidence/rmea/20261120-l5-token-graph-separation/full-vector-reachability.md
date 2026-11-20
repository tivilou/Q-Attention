# Label-blind full-vector mixed-query support diagnostic

For fixed unit keys let E_i=psi_i psi_i†+eps I and D=sum E_i=M+N eps I>0. For every physical mixed query rho>=0,tr rho=1, native probabilities are tr(E_i rho)/tr(D rho). With F_i=D^(-1/2) E_i D^(-1/2), the map rho -> D^(1/2)rho D^(1/2)/tr(D rho) is a bijection on density matrices. Equivalently, a Cholesky inverse L^(-1) with D=LL† produces unitarily equivalent effects F_i=L^(-1)E_i L^(-†). They are positive and sum to I.

For every real weight vector w the exact support function of the complete mixed-query probability region is

    sup_rho sum_i w_i p_i = lambda_max(sum_i w_i F_i).

Consequently, w·a greater than this eigenvalue strictly separates candidate probability vector a from every native pure or mixed query at those fixed keys. With ||w||_2=1, the excess also lower-bounds Euclidean distance to the region. For the initially-dephased comparator replace F_i by sum_j |U_ij|² F_j. Unitarity makes this another POVM with the same denominator.

The optimization is only a direction proposer. It minimizes half the squared probability residual over density matrices by 512 fixed accelerated projected-gradient steps, starting from I/8. The Lipschitz constant is the squared largest singular value of the Hermitian-coordinate measurement matrix. Each projection diagonalizes an 8x8 Hermitian matrix and projects its eigenvalues onto the probability simplex. The final residual normalized in l2 supplies w. Convergence is not assumed: every proposed real w has the same exact support bound. A missed witness is inconclusive, never a proof of attainability. The implementation subtracts the computed eigenpair backward residual and a small floating-point allowance; this is a complex128 numerical diagnostic, not interval-arithmetic certification.

This can detect a separation when no individual coordinate exceeds its native ceiling. It is stronger than comparing actual native and coherent outputs for the same query. It does not compare globally re-encoded/learned families or establish a relation-classification benefit.

The input set is fixed before measuring this diagnostic: the six cubic-condition final traces already materialized by the accepted technical parent canary, two hash-ordered units per train/valid/test split. That canary performed one update per head. These are authentic feature/circuit inputs with a very weak training budget, not a formal efficacy cohort. All forty queries are included. Gold labels, token text, predictions, utilities and outcomes are not accessed. Native, linear coherent and dephased outputs provide feasible-null controls. No scientific training, COS upload, new source-data acquisition or formal protocol change occurs.

Toy check: the middle-token point mass in the three-key construction violates the known native maximum (1+eps)/(2+3eps), independently of optimizer convergence. This checks the numerical support search, not the coherent witness or task utility.

## Conditional complete pure-query criterion (theory only)

Scope: exact finite-dimensional mathematical statements for the current frozen route, not a new method, empirical experiment, implementation review, genericity claim or L5 acceptance. Queries in the comparator range over every physical pure state; the actual depth-two query circuit is a subset. No priority claim is made for tomography, singular-value stability, or stochastic-matrix tools.

### 1. Actual normalization and informational completeness

For normalized keys psi_i in C^d, d=8, let A have rows psi_i^dagger, P_i=|psi_i><psi_i|, S=A^dagger A and B=S+N*eps*I with eps=1e-6>0. For normalized phi the native attention is (phi^dagger P_i phi+eps)/(phi^dagger B phi). The bijective ray change chi=B^(1/2)phi/sqrt(phi^dagger B phi) gives effects

    E_i=B^(-1/2)(P_i+eps*I)B^(-1/2), sum_i E_i=I,
    p_i=Tr(E_i rho), rho=|chi><chi|.

The inverse ray map is phi=B^(-1/2)chi/||B^(-1/2)chi||. Thus all physical pure queries are exactly the pure-state probability range of E, including the EPS floor. The real Hermitian span of E_i is full iff that of P_i is full: X->X+eps*Tr(X)*I is invertible (identity eigenvalue1+d*eps, traceless eigenvalue1), as is congruence by B^(-1/2). Full rank requires N>=d^2=64 but is not implied by this count. No assertion that a retained/future sample qualifies.

### 2. Complete pure-query membership with a distance bound

Let t_i=Tr(E_i)/d and F_i=E_i-t_i*I. On the real Hilbert space of traceless Hermitian matrices with Frobenius norm, define L(X)_i=Tr(F_i X). If E is informationally complete, L is injective. Let s_min>0 and s_max be its smallest/largest singular values. For any probability vector p, define

    q=p-t, Xhat=L^+q, rhohat=I/d+Xhat,
    r=q-L(Xhat), eta^2=Tr(rhohat^2)+1-2*lambda_max(rhohat).

All quantities are real and rhohat is Hermitian with trace1, without a positivity assumption. r is orthogonal to range(L). eta is exactly the Frobenius distance of rhohat to the set of pure density matrices, by maximizing <v|rhohat|v>. Consequently

    sqrt(||r||_2^2+s_min^2*eta^2)
      <= dist_2(p, native all-physical pure-query range)
      <= sqrt(||r||_2^2+s_max^2*eta^2).

Proof: for every pure rho, p-p(rho)=r+L(rhohat-rho); the summands are orthogonal, and singular-value bounds apply to the traceless matrix rhohat-rho. Minimize for the lower bound; for the upper bound use a top eigenvector projector of rhohat. Evaluating that particular projector gives a sharper computable upper witness. In exact arithmetic the lower bound is zero iff p is native-pure realizable, so the membership criterion is complete in this IC regime. A mixed positive rhohat with r=0 is mixed-simulable but still fails the pure-query criterion unless it has rank1.

If L is not injective, only ||r|| is retained as a universal lower bound against both pure and mixed native ranges; the min-norm inverse's purity defect is not a valid lower bound because an unobserved kernel component could change it. With full rank, the same argument gives a mixed-range lower bound by replacing eta with the distance from rhohat's eigenvalues to the probability simplex. That optional variant is not needed for a pure-only witness.

An upper witness concerns all physical queries; it need not be reachable by the actual finite query circuit. The lower bound remains valid for that smaller family. Numerical use requires controlled input/rounding errors and conditioning; a floating-point rank threshold or positive decimal alone is not an exact certificate. No numerical witness is asserted here.

### 3. The initial-dephased control preserves rank at actual tau

For any Hermitian H with ||H||op<=1, U=exp(-i*tau H), M_ij=|U_ij|^2 is doubly stochastic. For |tau|<pi/4,

    |U_ii|>=Re(U_ii)>=cos(|tau|), M_ii>=cos^2(tau),
    ||M-I||_infinity, ||M-I||_1 <=2*sin^2(tau),
    ||M-I||op<=2*sin^2(tau), s_min(M)>=cos(2*tau)>0.

The cosine bound follows from the spectral interval[-1,1]; row and column off-diagonal masses both equal1-M_ii. This proves nonsingularity without an empirical rank assumption. At tau=1/4 the singular-value lower bound is cos(1/2), about.87758. The actual cubic/linear normalized generators satisfy the norm assumption; a singleton has M=I.

The exact initially dephased attention is M*p_native, including EPS since M*1=1. Its effects D_i=sum_j M_ij E_j, center t_D=M*t and traceless analysis map L_D=M*L. Therefore D is IC iff E is IC, and

    s_min(L_D)>=cos(2*tau)*s_min(L),
    s_max(L_D)<=s_max(L).

The distance/membership construction applies separately to the native and dephased comparators. A cubic probability vector may be tested against both; native exclusion alone is not dephased exclusion. For linear coherent routing the previously proved common physical-query unitary commuting with S makes native membership exact, so its pure-range distance is0 in real arithmetic.

### 4. Claim boundaries and information interpretation

In the IC regime native probabilities already identify the entire query density matrix. Hence any other fixed-key POVM, including the coherent one, has an exact real linear reconstruction from native outcomes, and from dephased outcomes via M^(-1). The coefficients need not be nonnegative, so this is not necessarily a stochastic postprocessing, and it may require globally key-conditioned value transformations outside the frozen local native head. This blocks a claim that coherent routing creates query information absent from both IC controls; it does not erase its fixed-head probability-region differences or imply task/classical/hardware advantage.

Theorems about pure-region separation do not predict Macro-F1. Existing trained pilot evidence already shows fixed-head cubic CE improves by.00137203674 while positive39Macro falls by.00308293526. All39class contributions and formal task gates must be assessed separately. The final formal report is pending. No new fixture, seed/tau search, optimizer step, checkpoint download or frozen-contract change follows from this note.

The underlying IC dual-frame reconstruction is established prior mathematics: Roy and Scott, quant-ph/0703025v2, Section5.2 pp16-17 Eqs5.9-5.12. This conditional criterion is an alternative to the unresolved generic-disjointness argument, not its proof. Independent bounded mathematics review: gpt-6.2-astra, medium, response resp_09299de6faf0f4a5016b00d215eef887d1909949ebd3d639c4; PASS/no must-fix for EPS mapping, distance, dephased control and reconstruction. Frozen proposition SHA256 aa14af78875548b5fffcbd0915e782600e9fd5678e28f23694ee2f1bf8ac1042; review verdict SHA256 0e52d05706fd25b1b5c85a500f49c13e9a4cf22cb4e4748f6316aa3fdc58f6ce. This review does not cover an implementation or the separate global-rigidity lead. The existing registered numerical diagnostic and frozen formal contract are unchanged.

## Actual-circuit scope of the IC condition

The implementation does not structurally forbid IC key effects merely because one key uses12angles. This is a direct architectural derivation, not an observation about the trained corpus or an independently reviewed theorem. Source: baselines/retacred/rmea.py lines14-33,55-57 (SHA2569f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0).

Write one directed CNOT ring as the fixed unitaryW and one local layer asL. The actual circuit is W L2 W L1 |000>. Set every second-layer angle to0. It becomes W^2 applied to an arbitrary product of three one-qubit RZ(z)RY(y)|0> states. In each wire choose four first-layer (y,z) pairs

    (0,0), (pi/2,0), (pi/2,pi/2), (-pi/2,0).

Up to irrelevant phases these give |0>, |+>, |+i>, |->. Their projectors have unscaled Pauli coefficient rows (I,X,Y,Z)

    (1,0,0,1), (1,1,0,0), (1,0,1,0), (1,-1,0,0),

whose determinant is-2. Hence they span the four-dimensional one-qubit Hermitian space. The64three-wire tensor products span Herm(C^8). Conjugation by fixedW^2 is an invertible real-linear map on that space, so these64allowed circuit states have linearly independent projectors. All listed angles lie strictly inside(-pi,pi), as required by pi*tanh; no unattainable infinite preparation coefficient is used.

Consequently, for64independently variable angle arrays in that open box, the projector determinant is a nonzero real-analytic function. Its zero set has measure0. Thus generic independently varying allowed angles yield IC; this is not a theorem that actual corpus keys are independent, continuous, diverse enough, or well conditioned. For more than64keys, one full-rank64subset suffices; fewer than64cannot be IC. Coincident/near-coincident prepared keys can still be rank-deficient/ill-conditioned regardless of circuit capacity.

The learned shared preparation does not automatically make arbitrary angles available. A sufficient continuum-capacity condition is full row rank12of the12-by1024 feature block of prepare.weight with roles held fixed: affine preimages of every finite inverse-tanh angle array then exist in unrestricted real feature space. No such unrestricted-feature assumption is justified for the fixed RoBERTa corpus, and neither trained weight rank nor per-sentence projector rank is asserted here.

Finally, full operator span is different from covering every pure query: the depth-two query circuit has12real parameters whereas CP^7has real dimension14. The IC distance theorem above intentionally uses all physical pure queries as an outer comparator. A positive controlled lower bound excludes the actual circuit family too; a zero bound only establishes all-physical membership and does not establish finite-circuit attainability. This capacity check closes a possible architectural obstruction. It adds no task gain, generic native/coherent disjointness, new fixture, source change or training request. Further empirical interpretation waits for the already frozen full-data report and its approved final-state traces.
