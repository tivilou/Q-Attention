# Label-blind full-vector mixed-query support diagnostic

For fixed unit keys let E_i=psi_i psi_i†+eps I and D=sum E_i=M+N eps I>0. For every physical mixed query rho>=0,tr rho=1, native probabilities are tr(E_i rho)/tr(D rho). With F_i=D^(-1/2) E_i D^(-1/2), the map rho -> D^(1/2)rho D^(1/2)/tr(D rho) is a bijection on density matrices. Equivalently, a Cholesky inverse L^(-1) with D=LL† produces unitarily equivalent effects F_i=L^(-1)E_i L^(-†). They are positive and sum to I.

For every real weight vector w the exact support function of the complete mixed-query probability region is

    sup_rho sum_i w_i p_i = lambda_max(sum_i w_i F_i).

Consequently, w·a greater than this eigenvalue strictly separates candidate probability vector a from every native pure or mixed query at those fixed keys. With ||w||_2=1, the excess also lower-bounds Euclidean distance to the region. For the initially-dephased comparator replace F_i by sum_j |U_ij|² F_j. Unitarity makes this another POVM with the same denominator.

The optimization is only a direction proposer. It minimizes half the squared probability residual over density matrices by 512 fixed accelerated projected-gradient steps, starting from I/8. The Lipschitz constant is the squared largest singular value of the Hermitian-coordinate measurement matrix. Each projection diagonalizes an 8x8 Hermitian matrix and projects its eigenvalues onto the probability simplex. The final residual normalized in l2 supplies w. Convergence is not assumed: every proposed real w has the same exact support bound. A missed witness is inconclusive, never a proof of attainability. The implementation subtracts the computed eigenpair backward residual and a small floating-point allowance; this is a complex128 numerical diagnostic, not interval-arithmetic certification.

This can detect a separation when no individual coordinate exceeds its native ceiling. It is stronger than comparing actual native and coherent outputs for the same query. It does not compare globally re-encoded/learned families or establish a relation-classification benefit.

The input set is fixed before measuring this diagnostic: the six cubic-condition final traces already materialized by the accepted technical parent canary, two hash-ordered units per train/valid/test split. That canary performed one update per head. These are authentic feature/circuit inputs with a very weak training budget, not a formal efficacy cohort. All forty queries are included. Gold labels, token text, predictions, utilities and outcomes are not accessed. Native, linear coherent and dephased outputs provide feasible-null controls. No scientific training, COS upload, new source-data acquisition or formal protocol change occurs.

Toy check: the middle-token point mass in the three-key construction violates the known native maximum (1+eps)/(2+3eps), independently of optimizer convergence. This checks the numerical support search, not the coherent witness or task utility.
