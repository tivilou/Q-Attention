# Signal-relative nuisance stability for the unchanged cubic route

Prospective theorem and fixed-state task prediction, not a method change, new experiment, novelty clearance or L5 acceptance. Source: clean development65d24c0329225e893c2eeaf459ee7a07b327d75f, unchanged baselines/retacred/rmea_token_transport.py.

## Exact implemented interface
For normalized keys psi_i in C^d (actual d=8), A has rows psi_i^dagger, G=AA^dagger and amplitudes a=A phi_r. Class queries have norm1, utilities u_ri are real and fixed for each input feature row; bias b_r is fixed. For q in {1,3}, set K_q offdiagonal entries G_ij |G_ij|^(q-1), diagonal0, H_q=K_q/||K_q||op and U_q=exp(-i tau H_q), tau=1/4. Coherent alpha_ri=(|(U_q a)_i|^2+epsilon)/(||a||^2+N epsilon), epsilon=1e-6. The actual initially-dephased cubic control is alpha_native @ |U_3|^2^T; equivalently diagonal of U_3 diag(|a|^2+epsilon) U_3^dagger divided by the same denominator.

## Comparison and assumptions
Partition one frozen-feature input into retained R, |R|=r>=2, and nuisance B, |B|=m>=1. Removing B means masking its rows only at the head; no contextual encoder recomputation or semantic equivalence is assumed. Queries, retained states, retained utilities and bias remain fixed. Let K_R,q be the retained principal graph, lambda_q=||K_R,q||op>0. Embed it as K_0,q=diag(K_R,q,0_B), and E_q=K_q-K_0,q, e_q=||E_q||op. Let S_R,r=||a_R||^2, S_B,r=||a_B||^2 and

 delta_r=(S_B,r+m epsilon)/(S_R,r+S_B,r+(r+m)epsilon).

This is the exact native attention mass of B, including its EPS floor. Let Delta_r=max_(R union B) u_ri-min_(R union B)u_ri. No small-query-mass conclusion is inferred from key-key geometry.

## Candidate finite-tau stability bound
For coherent q=1 or3, and the initially-dephased control with its matching q, compare z_full,q,r with z_R,q,r, each evaluated under the same frozen query/utilities and its exact own native denominator and own operator-normalized graph. Proposed bound:

 |z_full,q,r-z_R,q,r| <= Delta_r [delta_r+2 |tau| e_q/lambda_q].    (1)

The bias cancels. This is individual-route stability relative to its own retained-token output, not a cubic-versus-linear task-utility order. One may also cap a valid result by Delta_r.

Proposed proof: principal-submatrix norm monotonicity gives Lambda_q=||K_q||op>=lambda_q>0, and reverse triangle gives |Lambda_q-lambda_q|<=e_q. Therefore

 ||K_q/Lambda_q-K_0,q/lambda_q||op<=2e_q/Lambda_q<=2e_q/lambda_q.

Duhamel's formula gives ||U_q-U_0,q||op<=2|tau|e_q/lambda_q. Under coherent evolution, the smoothed probabilities are the diagonal of U (a a^dagger+epsilon I) U^dagger divided by D. The epsilon I cancels between the two unitaries; probability L1 difference is at most2||a||^2||U-U0||op/D<=4|tau|e/lambda. For an initially dephased state the same L1 bound <=2||U-U0||op follows from trace distance of normalized density matrices (the identity part again need not change). Any real utility has expectation difference at most Delta_r/2 times that L1 distance. Under U0, the full retained contribution is exactly (1-delta_r) z_R without bias, and nuisance output is delta_r times the native nuisance utility mean. Their pooled expectation differs from retained-only by at most delta_r Delta_r. Triangle inequality yields (1).

This proof is for exact arithmetic. Near-zero lambda may make the bound vacuous; the implementation rejects zero multi-token graphs. It does not assert smooth differentiation at norm ties.

## Key-affinity sufficient condition and cubic-specific prediction
Suppose every offdiagonal overlap touching B, including B-B, has |G_ij|<=eta, and at least one retained R-R pair has overlap magnitude>=rho>0. Then

 e_q <= [sqrt(rm)+(m-1)] eta^q,
 lambda_q >= rho^q,
 e_q/lambda_q <= [sqrt(rm)+(m-1)](eta/rho)^q.               (2)

Offdiagonal cross-block [[0,C],[C^dagger,0]] has norm||C||<=sqrt(rm)eta^q; the B-B block has norm<=(m-1)eta^q by absolute row sums. A retained2x2 principal block gives lambda>=rho^q.

For a true signal/noise contrast eta<rho, the transport-sensitive term in (1) thus scales at most cubically rather than linearly in eta/rho, while the Born/EPS nuisance mass delta is shared and need not be small. This rigorously isolates one possible advantage of cubic affinity suppression under explicit graph-separation and query-mass assumptions; it does not prove typical NLP keys satisfy them or that cubic actual error is always lower.

For a fixed multiclass example with target y, define C_r=Delta_r min(1,delta_r+2|tau| e_q/lambda_q). From coordinate bounds, the exact cross-entropy perturbation obeys

 |CE(z_full,y)-CE(z_R,y)| <= C_y+max_r C_r <=2 max_r C_r.      (3)

For a retained-only predicted class k with positive margin z_R,k-max_(r!=k)z_R,r, the full prediction is unchanged if that margin > C_k+max_(r!=k)C_r. This is a robustness guarantee, not a gain certificate or a claim about the gold class being correct.

## Falsifiable use, limits and test boundary
A useful prediction concerns fixed-state sensitivity to a label-blind predeclared token mask, measured on the same original examples and with no representation recomputation, training, seed/tau/threshold search or outcome selection. First measure whether graph separation, e_q/lambda_q, delta_r and nonvacuous output/margin bounds occur in the existing approved real-example traces. A high delta, weak anchor lambda or strongly connected nuisance block defeats the proposed explanation even if the geometric theorem is true. Only then could a separately frozen robustness intervention compare route sensitivity; masking is a head intervention, not a semantic-invariance certificate.

The bound also applies to the initially-dephased graph control, so it cannot establish coherence necessity. Existing independent formal four-route Macro/CE gates remain the utility test. Known power-kernel suppression, operator perturbation and Duhamel/trace-distance inequalities are standard ingredients. Even a correct bound is not by itself L5 or a novelty claim; relevant graph stability originals must be compared. No new scientific execution is proposed by this analytical review. Review only the formulas, constants, exact smoothing/control correspondence and scope. Do not request a lifecycle audit, whole suite or extra geometry-only fixture.
