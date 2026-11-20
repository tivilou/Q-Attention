# Prospective full-rank, overcomplete and perturbation-stable separation

Status: Astra6.2 medium analytical review PASS_WITH_RISK (resp_078a9a5a477ff31c016b0095d9aa4487d2a97599a896c760ec). Its required zero-perturbation endpoint notation correction is applied below; all three affected bounds are non-strict. No new numerical measurement or task-benefit claim. The released protocol, source and parameters are unchanged.

The existing three-key construction has rank two. With zero smoothing, linearly independent N<=d keys can realize every attention probability vector by changing the physical query, so a rank-deficient triple alone leaves a serious genericity question. The following extension uses N=9,d=8 and a full-rank frame with smallest frame eigenvalue one.

## Construction at the exact source constants

Take an orthonormal basis e_0,...,e_7 of C^8. In token order, set

    psi=(e_0, v, e_1, e_2,...,e_7), v=a e_0+b e_1,
    a=sqrt(3)/2, b=1/2, tau=1/4, epsilon=10^-6.

Thus N=9 and M=Psi Psi^dagger=I_8+vv^dagger, with eigenvalues 2,1,...,1. The cubic off-diagonal graph K_ij=G_ij|G_ij|^2 (G=Psi^dagger Psi) consists of the established three-node path and six isolated vertices. Its norm is c_0=sqrt(7)/4. Put H=K/c_0 and U=exp(-i tau H), precisely as in the released route().

Write r=3sqrt(3)/(2sqrt(7)), s=1/(2sqrt(7)), x=r e_0+s e_1, q=sin(tau)^2, w=cos(tau)v+i sin(tau)x. Set

    ell=w^dagger M^-1 w=1/2+3q/56,
    beta=||M^-1 w||^2=1/4+9q/112,
    phi_star=M^-1 w/sqrt(beta).

Only the normalization changes from the three-token proof. The coherent middle probability is

    P=(ell^2+epsilon beta)/(ell+9epsilon beta).

## Every native or initially-dephased physical query obeys the same ceiling

For any unit physical query, and any trace-one mixed physical state,

    p_native_middle <= L=(1+epsilon)/(2+9epsilon).

This is the largest generalized eigenvalue of vv^dagger+epsilon I versus M+9epsilon I: the ratio along v is L and every orthogonal direction has ratio epsilon/(1+9epsilon).

The initially-dephased middle effect before smoothing is

    F=(1-q)vv^dagger+q D,
    D=r^2 e_0e_0^dagger+s^2 e_1e_1^dagger.

On span{e_0,e_1},

    L(M+9epsilon I)-(F+epsilon I)
      =gamma(I-vv^dagger)+q(vv^dagger-D),
    gamma=(1+8epsilon)/(2+9epsilon)>=1/2.

On its six-dimensional orthogonal complement the difference is gamma I. In the {v,v_perp} basis, vv^dagger-D has diagonal entries 15/56,-15/56 and off-diagonal magnitude 13sqrt(3)/56. The two-dimensional determinant is q(210gamma-183q)/784. Hence it is positive semidefinite for 0<=q<=35/61, which includes tau=1/4. Therefore p_initial_dephased_middle<=L for every physical query, including mixed states.

Put z=3q/56. Exact cross multiplication gives

    (2+9epsilon)(ell^2+epsilon beta)-(1+epsilon)(ell+9epsilon beta)
      =z(1+2z)+epsilon z(-5/2+9z)>0

at epsilon=10^-6. Thus P>L. A deliberately conservative bound is P-L>0.003: Taylor bounds imply 0.06<q<1/16, hence z>0.18/56, ell<0.504, beta<0.256, the displayed numerator is >z, and the positive cross-multiplication denominator (ell+9epsilon beta)(2+9epsilon) is <1.009. These bounds alone give >0.003.

This comparison fixes the keys and readout coordinate while allowing every competing physical query. It is neither a global learned-family expressivity theorem nor a task-utility result.

## Explicit open neighborhood: max key perturbation <=10^-5

Let all nine perturbed keys remain unit norm and max_j||psi'_j-psi_j||_2<=dlt, where dlt<=10^-5. Let A=Psi^dagger, A'=Psi'^dagger, and use the same cubic generator normalization and fixed tau/EPS to form U'. All comparisons below are uniform in unit physical queries.

1. ||A'-A||op<=3dlt. Each Gram-entry difference is <=2dlt. The map z->z|z|^2 is 3-Lipschitz on the unit disk, so ||K'-K||op<=48dlt (maximum absolute row sum of the Hermitian 9x9 off-diagonal difference).
2. Since ||K||op=c_0>0.66 and K'!=0 in this neighborhood,

       ||K'/||K'||-K/c_0||op<=2||K'-K||op/c_0<=146dlt.

   Duhamel's formula for Hermitian exponentials gives u=||U'-U||op<=36.5dlt. This uses norm Lipschitz bounds, not differentiability at the tied spectral norm.
3. Let mu=3dlt(2sqrt(2)+3dlt). Then ||M'-M||op<=mu<=8.5dlt+9dlt^2. All probability denominators are >=b-mu, b=1+9epsilon, and b-mu>0.9999. In particular M' remains positive definite, so the perturbed frame still spans C^8.
4. The transported middle row changes by at most eta=3dlt+sqrt(2)u<=55dlt; its unperturbed norm is one. The coherent middle-probability change at the fixed query phi_star is bounded by

       Cc=[eta(2+eta)+mu]/(b-mu)<0.001186.

   Here the exact ratio perturbation bound uses the unperturbed probability <=1.
5. Native middle and every other native-coordinate probability change uniformly by at most

       Cn=[dlt(2+dlt)+mu]/(b-mu)<0.000106.

   For initially-dephased probabilities T p_native with T_ij=|U_ij|^2, each row of T is stochastic and its row-l1 change is <=2||U'-U||op<=2u. Therefore the middle initially-dephased probability changes by at most

       Cd=Cn+2u<0.000836.

Consequently the perturbed coherent probability at the same fixed phi_star exceeds the supremum middle probability of both perturbed native and perturbed initially-dephased routes by

    (P-L)-Cc-max(Cn,Cd)>0.003-0.001186-0.000836>0.0009.

This supplies an explicit open, full-rank, overcomplete family, rather than only a rank-two triple. Linear-Gram transport remains query-absorbed for every perturbed key set by the already established push-through identity.

## Limits and prospective prediction

The radius is a conservative mathematical sufficient condition, not a measured robustness radius, typicality claim, input/encoder reachability claim, or task robustness result. The six initially isolated vertices need not stay isolated: the perturbation statement permits all cross-block overlaps to become nonzero. The original three-key subset can still have probability-region behavior different from a generic independent triple. No novelty claim follows without frame/measurement and quantum-walk literature comparison.

Prediction: under the stated normalized key perturbations, the fixed middle coordinate remains above both query-optimized competing ceilings; the linear generator never expands the entire native query region. A future bounded numerical test should use the released exact route and generalized-eigenvalue ceilings, not sampled-query maxima. It would verify this same frozen mechanism, not modify the formal experiment or rescue a failed task result.
