# Candidate: a fixed-key attention ceiling separated by coherent token transport

Status: candidate theory; core equations, finite circuit mapping, asymmetric-family extension and linear-Gram null passed independent Astra6.2 medium analytical review. Reviewer terminology correction applied. New source numerical check pending.
This is a probability-level claim with fixed keys, not global learned-family expressivity, task benefit, quantum speedup, novelty clearance or L5 acceptance.

## Mechanism and scope
For normalized physical keys psi_t in C^8, Psi=[psi_0,...,psi_(N-1)], normalized query phi, EPS=epsilon>0,
G=Psi^dagger Psi, K_ts=G_ts|G_ts|^2 (t!=s), K_tt=0, H=K/||K||op (K!=0), U=exp(-i tau H).
Native p_t=(|psi_t^dagger phi|^2+epsilon)/(phi^dagger M phi+N epsilon), M=Psi Psi^dagger.
Coherent p_t=(|(U Psi^dagger phi)_t|^2+epsilon)/(phi^dagger M phi+N epsilon).
Initial-token-dephased p_t=sum_s |U_ts|^2 p_native_s. This retains U; it is not a continuous classical Markov walk or physical-basis dephasing.
Code: token_transport.py at source52370b2741f4ae8ac3d6b88e10b09ac87794cf45, forward-only, tau=1/4, epsilon=1e-6.

## Exact three-key theorem
Choose an orthonormal physical pair e,f in C^8 and keys psi_0=e, psi_1=v=a e+b f, psi_2=f,
a=sqrt(3)/2, b=1/2. M=I_2+vv^dagger on their span and zero outside.
Raw K path edges are 3sqrt(3)/8 and 1/8, norm sqrt(7)/4. Hence
H=[[0,r,0],[r,0,s],[0,s,0]], r=3sqrt(3)/(2sqrt(7)), s=1/(2sqrt(7)), r^2=27/28, s^2=1/28, H^3=H.
Let q=sin(tau)^2, x=r e+s f, and w=cos(tau)v+i sin(tau)x.
The coherent middle coordinate has effect ww^dagger: (U Psi^dagger phi)_1=w^dagger phi.
Use normalized query phi_star=(I_2-vv^dagger/2)w/sqrt(beta), with
ell=w^dagger M^-1 w=1/2+3q/56 and beta=||M^-1 w||^2=1/4+9q/112.
Here M^-1 denotes the inverse on the two-dimensional key span only.
Its coherent middle weight is P=(ell^2+epsilon beta)/(ell+3epsilon beta).

For every normalized pure query in C^8, and even for every trace-one density matrix rho,
p_native_1 <= L=(1+epsilon)/(2+3epsilon).
Proof: maximize the generalized Rayleigh quotient of A=vv^dagger+epsilon I_8 and B=M+3epsilon I_8.
Their eigenvalue ratios are L along v, epsilon/(1+3epsilon) on its orthogonal direction in the span, and 1/3 outside the span; L is largest. The trace quotient for mixed rho obeys the same Loewner bound A<=LB.

The initial-dephased middle effect is F=(1-q)vv^dagger+q D, D=diag(r^2,s^2) on span{e,f}.
For gamma=(1+2epsilon)/(2+3epsilon), the difference L(M+3epsilon I)-(F+epsilon I) equals
gamma |v_perp><v_perp|+q(vv^dagger-D) on the span, and epsilon/(2+3epsilon) I outside.
In basis {v,v_perp}, vv^dagger-D has entries [[15/56,13sqrt(3)/56],[13sqrt(3)/56,-15/56]]
(the off-diagonal sign is basis-dependent). Its determinant after adding gamma is q(210gamma-183q)/784.
It is positive semidefinite for 0<=q<=70gamma/61; since gamma>=1/2, q<=35/61 is sufficient for every epsilon>=0.
Thus p_initial_dephased_1 <= L for every physical query under that condition.

Write delta=3q/56>0. The cross-multiplied numerator of P-L is
(2+3epsilon)(ell^2+epsilon beta)-(1+epsilon)(ell+3epsilon beta)
=delta(1+2delta)+epsilon delta(1/2+3delta)>0.
Therefore, for 0<sin(tau)^2<=35/61, P>L>=max(p_native_1,p_initial_dephased_1) for all competing queries.
The fixed source tau=1/4 qualifies since sin(1/4)^2<1/16<35/61. EPS is retained, not sent to zero.
The epsilon=0 extension requires nonzero denominators; the source and main theorem use epsilon>0.

## Finite depth-two native-circuit realization, proposed mapping
Do not use an infinite tanh weight to prepare a pi rotation. Let physical basis b0=|000>, b3=|011>.
Use e=(b0-b3)/sqrt(2), f=(b0+b3)/sqrt(2). First-layer rotations all zero.
In the last layer, only wire0 RY is nonzero; its CNOT ring maps |100> to |011>.
Token RY angles (-pi/2,-pi/6,+pi/2) yield e, (sqrt(3)/2)e+(1/2)f, f exactly.
Use RMEAHead/TokenTransportHead.double(), float64 features and roles, boolean mask with all three tokens valid. circuit() then produces complex128. Exact equality refers to real arithmetic; executed float64 uses tolerances. Choose three synthetic feature rows h_t as standard basis rows 0,1,2 in R^1024; all role features zero.
Set prepare weights/bias zero except flattened last-layer-wire0-RY row6 columns0,1,2:
atanh(-1/2), atanh(-1/6), atanh(1/2), respectively. All are finite.
To realize phi_star, write its normalized physical coefficients as A b0+B b3,
where z=M^-1 w has coordinates (z_e,z_f), A=(z_e+z_f)/sqrt(2 beta), B=(-z_e+z_f)/sqrt(2 beta).
Choose last-layer-wire0 RY y=2 atan2(|B|,|A|), RZ zeta=Arg(B)-Arg(A), wrapped to (-pi,pi).
Set effect[1,1,0,0]=atanh(y/pi), effect[1,1,0,1]=atanh(zeta/pi), all other effect angles zero.
This prepares phi_star up to global phase provided |y|,|zeta|<pi; analytically expected at tau1/4, to be checked independently and by the source canary.
No shared additional circuit or extra depth is introduced. The common rotation above is a choice of mathematical basis, not an additional implemented gate.

Optional finite decision witness: keep all value/decoder/bias entries zero except value.weight[0,1]=1,
decoder[1,0]=1, bias[0]=(P+L)/2 and bias[c]=-2 for c>=2; bias[1]=0.
Then class1 utilities over three feature rows are (0,1,0), class0 logit is the threshold, all other logits -2.
Native and initial-dephased predict class0 for any class1 query; coherent at phi_star predicts class1.
Relabelling the same synthetic example class0 reverses which behavior is useful. This is not evidence of text benefit.

## General measurement for future task-linked testing
For any fixed sample, the exact native ceiling is lambda_max(B^-1/2 A_t B^-1/2),
B=M+N epsilon I, A_t=psi_t psi_t^dagger+epsilon I.
The initial-dephased ceiling replaces A_t by sum_s |U_ts|^2 psi_s psi_s^dagger+epsilon I.
Comparing actual coherent attention with both ceilings can certify fixed-key/query-only nonabsorption on real samples.
Positive ceiling excess need not be useful: utility sign, rivals and CE must be measured on the same fixed example.
No-excess does not imply identical attention families or disprove all coherent utility; this coordinate certificate is sufficient, not necessary.
The next empirical design should connect any observed excess to unchanged utility readout and matched decisions, and separately test ordinary held-out utility under frozen training budgets.

## Essential limitations and nearest-work boundary
Exact contextual-key classical counterpart remains the effective synthesis (key-column) matrix Psi_tilde=Psi U^dagger; its adjoint U Psi^dagger is the analysis matrix acting on a query. Its columns need not be unit physical keys. Their squared norms are diag(U G U^dagger); independently normalizing them can change attention. The theorem holds original normalized keys fixed.
Retraining/changing keys, encoder, values or arbitrary contextual classical routing is outside its comparator class.
The generalized-Rayleigh/Loewner tools are known mathematics; no novelty claim for those identities.
Quantum-walk neural propagation (Dernbach et al.2018, arxiv1801.05417v1), complex magnetic graph spectral filtering (Zhang et al.2021, arxiv2102.11391v1) and phase-aware factorized attention (Nahid et al.2026, arxiv2608.17288v1) are known neighbors.
The current proof is a narrowly specified theorem about this cubic-Gram graph and fixed-key attention region; independent literature priority remains unresolved.
Existing forward canary8/8 does not contain this new finite three-key ceiling theorem, gradients, CUDA execution or real-data utility.
L5 NOT_ACCEPTED. Historical scoped L4 and every stopped family remain unchanged.

## Generalized asymmetric-family extension (new in v002)
For any a,b>0 with a^2+b^2=1 and a!=b, use keys e,v=a e+b f,f and the same cubic-Gram construction.
Put Z=a^6+b^6, r=a^3/sqrt(Z), s=b^3/sqrt(Z), x=r e+s f,
h=v^dagger x=(a^4+b^4)/sqrt(Z), q=sin(tau)^2.
Strict Cauchy-Schwarz gives 0<h<1 precisely for this positive unequal interior family.
The same witness phi_star=M^-1 w/||M^-1 w||, w=cos(tau)v+i sin(tau)x, obeys
ell=1/2+delta, beta=1/4+3delta/2, delta=(1-h^2)q/2.
Thus the same smoothed P-L numerator is delta(1+2delta)+epsilon delta(1/2+3delta)>0 for q>0.
Define kappa=a^2 s^2+b^2 r^2>0 and xi=kappa-r^2 s^2=(a^2-r^2)^2+a^2 b^2>0.
For the initial-dephased effect, the span slack has trace gamma=(1+2epsilon)/(2+3epsilon) and determinant
gamma kappa q-xi q^2. Hence both comparator ceilings remain <=L whenever
0<q<=min(1,gamma kappa/xi); the epsilon-uniform sufficient interval is 0<q<=min(1,kappa/(2xi)).
This produces an open family of coherent-vs-both-comparator coordinate separations, rather than a lone parameter witness.
At a=b=1/sqrt(2), x=v, delta=0, and this middle-coordinate excess vanishes exactly. Endpoints a=0 or b=0 are excluded.
At a=sqrt(3)/2,b=1/2, kappa=15/56 and xi=183/784, recovering 35/61.
This extension is an abstract family theorem; the finite native realization is currently proposed only for the original a=sqrt(3)/2 case. It is not an encoder reachability statement.

## Linear-Gram null and design distinction (new in v002)
For arbitrary normalized fixed keys, the linear off-diagonal Gram generator is K_lin=G-I_N.
If c=||K_lin||op>0, set U_lin=exp[-i tau(G-I_N)/c] and W=exp[-i tau(M-I_8)/c].
The push-through identity (G-I_N)Psi^dagger=Psi^dagger(M-I_8) implies U_lin Psi^dagger=Psi^dagger W.
W is unitary and commutes with M, so coherent linear-Gram attention at phi equals original-key native attention at W phi, including epsilon and normalization.
As W is bijective on physical queries, the complete fixed-key attainable attention sets are equal; every native coordinate ceiling stays unchanged. K_lin=0 gives the identity case.
This comparator allows all physical queries and says nothing about simultaneous global learned-query realization over a dataset, or whether a restricted circuit can implement each W.
Thus a nonlinear graph can be necessary for this particular fixed-key-region escape relative to the linear Gram route, but cubic is not uniquely necessary and necessity of all nonlinear constructions is not claimed.
The prospective controls are: asymmetric cubic coherent separation; its initial-dephased ceiling; balanced cubic no middle excess; linear-Gram no region expansion.
These are mathematical mechanism predictions. Their task relevance still requires actual fixed-example utility and held-out evidence.


## Minimal nonlinear edge family: reviewed refinement (2026-11-21)

Status: independently reviewed PASS_WITH_RISK by gpt-6.2-astra, medium, response resp_0f41f0c0e176bc6f016b007683bbd887d291939f4319593c5a. The review found no required mathematical correction within the stated scope. It does not change the completed four-condition screen. Literature priority and task benefit remain unresolved. The verdict is retained in theory-refinement-review.json.

Let the off-diagonal generator be entrywise K_ij=F(G_ij), with zero diagonal. Restrict F to a scalar polynomial in z and conjugate(z), require Hermiticity F(conjugate(z))=conjugate(F(z)), and local-key phase covariance F(exp(i theta)z)=exp(i theta)F(z). This deliberately narrow class excludes multi-edge/contextual graph functions.

**Polynomial classification.** Expand F(z)=sum_(p,q) c_pq z^p conjugate(z)^q. Covariance for every theta forces p-q=1. Hermiticity then forces the surviving coefficients to be real. Consequently F(z)=z h(|z|^2), where h is a real polynomial. Nonzero degrees are odd. Degree one gives only F(z)=a z; degree three is the first possible nonlinear member, a z+b z|z|^2. This is an elementary classification within the stated class, not a new general representation theorem.

**Why the linear member is query-only null.** For normalized keys and real a, K=a(G-I_N). Write S=Psi Psi^dagger and s=tau a/||K||op for K nonzero. Then
exp(-i s(G-I_N)) Psi^dagger = Psi^dagger exp(-i s(S-I_d)).
The right-hand physical operator V is unitary and commutes with S. Thus both the amplitude vector and its EPS-normalized denominator agree with native attention evaluated at V phi. V is invertible on all unit queries, so the entire fixed-key attainable probability region is unchanged. This does not assert equality of a restricted circuit ansatz across all samples, nor equal training outcomes.

**Two tokens are insufficient in this class.** For N=2 there is one off-diagonal magnitude, hence K=h(|G_12|^2)(G-I_2). It reduces to the linear-null case for each fixed key pair. N=1 is also null. More generally, if all nonzero off-diagonal overlaps have one common magnitude, the same reduction applies for any N. Zero generators require an explicit identity convention or rejection; the current normalized source rejects the multi-token zero case.

The already established asymmetric three-key cubic witness separates the coherent probability ceiling at fixed tau=1/4 and EPS=1e-6. Taken together, the results establish minimal polynomial degree three and minimal token count three for the *possibility* of nonabsorption in this entrywise phase-covariant family. They do not say every cubic graph or every triple separates, and they do not establish global learned-family expressivity or utility.

**The cubic kernel has a known feature decomposition.** Let chi(psi)=psi tensor psi tensor conjugate(psi), in Sym^2(C^d) tensor conjugate(C^d). Its Gram matrix is G^(3)=G elementwise |G|^2. On symmetric tensors define (CT)_a=sum_b T_abbar(b). Its adjoint is (C^dagger v)_abbar(c)=(v_a delta_bc+v_b delta_ac)/2, so C C^dagger=(d+1)I_d/2 and C chi(psi)=psi. Orthogonal projection onto range(C^dagger) therefore gives
G^(3)=2G/(d+1)+H, with H positive semidefinite.
H is the Gram matrix of the component in ker(C); its diagonal is (d-1)/(d+1). For d=8 the full symmetric cubic feature dimension is288 and the contraction-free ambient dimension is280; the finite-sample Gram need not have rank280.

After removing the diagonal,
K_cubic=2(G-I_N)/(d+1)+H-(d-1)I_N/(d+1).
This identifies an interpretable nonlinear component beyond the query-null linear member. The two summands need not commute, so one must not drop the linear term and claim identical dynamics; operator normalization also changes. Testing a contraction-free generator would be a distinct prospective hypothesis, not an allowed rescue of the current screen. The feature/projection identities themselves are standard algebraic tools and need near-neighbor attribution before any novelty claim.

Empirical implication for the current screen: linear is a structural comparator, while coherent/dephased cubic isolates the role of complex amplitudes for this fixed generator. A positive four-condition task gate would justify formal collaborator validation of that conjunction. A geometric ceiling witness alone cannot do so.

## Unitary absorption and a prospective label-blind diagnostic

The same independent review verified the following criterion, including rank-deficient keys. Put A=Psi^dagger, so G=AA^dagger. For a fixed token unitary U, a single physical query unitary V satisfying UA=AV exists if and only if UGU^dagger=G. Necessity follows by multiplying by the adjoint. Conversely equal Gram matrices define an isometry on the range of A^dagger, which extends to a unitary on the full physical space. No invertibility of A is assumed.

For U=exp(-i tau H), Hermitian H of operator norm1 and tau=1/4, the eigenvalue interval has no exponential phase aliasing. Thus [U,G]=0 if and only if [H,G]=0. This is an amplitude criterion for one common physical unitary, not a characterization of normalized probability regions or a theorem about learned task performance.

Token phases are invisible to Born probabilities. Allowing a fixed diagonal token unitary D changes the criterion to DUGU^dagger D^dagger=G. A nonzero ordinary commutator alone therefore does not exclude query absorption up to those phases. A sufficient, cheap certificate is

    row_norm_gap = max_t abs(sum_d abs((UA)_td)^2 - sum_d abs(A_td)^2).

Diagonal conjugation preserves the two diagonals, so a gap exceeding the declared numerical tolerance rules out every such D. Linear transport supplies the zero-residual control. A null result is inconclusive; even a positive result does not establish a coordinate probability-ceiling violation, coherent necessity, or utility.

Prospective full-data measurement: use the first32 outcome-blind hash-ordered test units, float64/complex128, absolute witness tolerance1e-8, and report residuals for all four routes. For coherent coordinates exceeding both native and initially dephased ceilings by more than1e-8, report the token/class identity privately, unchanged value contribution, true-versus-rival margin and same-example CE under all fixed-state swaps. Report the total inspected coordinates and zero-witness denominator when appropriate. These diagnostics do not affect the predeclared positive39Macro+CE task gate. No additional measurement is asserted for the completed screen, whose final models were not retained.
