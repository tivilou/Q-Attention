# Mass/damping cap and a finite-transmission certificate

Status: primary derivation submitted for independent checking. This extends cancellation-boundary.md; no training, experiment configuration or scientific source is changed. All claims below are conditional geometry, not literature priority, task utility or L5 acceptance.

## Assumptions

Real E/e, eta>0, gamma>0, normalized unit states in C^d (d>=2), q_i>0, sum q_i=1. Distinct s/o terminals. For nonterminals let rho=sum q_i, u_i=E-e_i+q_i, a_i=q_i/(u_i+i eta). Write C=I-sum a_i psi_i psi_i^H. The fixed-coefficient cancellation threshold c_star is defined and derived in the preceding note; it maximizes beta/sqrt(|alpha|²+beta²) with alpha=1-sum a_i t_i and beta=sum |a_i|sqrt(t_i(1-t_i)).

## Universal sharp cap

For any fixed coefficients,

c_star <= rho/sqrt(rho²+4 eta²).

The cap is sharp when both context state directions and real onsite values are free. At the actual E0, the attaining onsite values below lie inside(-1,1), but joint realizability through the actual trained feature/circuit/Born map is still unproved. The inequality applies without assuming that attainability.

Proof. Let p=sum q_i t_i and alpha=x+i y. Then y=sum eta*q_i*t_i/(u_i²+eta²)>=0. Two weighted Cauchy–Schwarz inequalities give

beta² <= (rho-p)y/eta,
|1-alpha|² <= p*y/eta.

Adding yields beta² <= rho*y/eta-|1-alpha|². Put r0=rho/(2 eta). Completing a square,

r0²|alpha|²-[2r0*y-|1-alpha|²]
=(1+r0²)[(x-1/(1+r0²))²+(y-r0/(1+r0²))²]>=0.

Therefore beta²<=r0²|alpha|² and the claimed cap follows, including p=y=0.

For sharpness at rho>0, set every nonterminal u_i=rho/2, hence e_i=E+q_i-rho/2. All a_i have the same phase and sum a=rho/(rho/2+i eta). Set t_i=1/2 and use the common-plane construction from the preceding note; all context states coincide and |1-a|=1. Then c_star=|a|/2=rho/sqrt(rho²+4 eta²), and the construction has exact zero T at this overlap. At E0, each e_i lies between-rho/2 and rho/2, so the scalar tanh range is respected. For rho0 there are no nonterminals and the cap is0. A witness in the free-state relaxation is not an actual learned-model witness.

Equivalently, an endpoint overlap c>0 requires at least rho>=2 eta*c/sqrt(1-c²) to permit exact cancellation. The right side may exceed the available mass; this is an impossibility condition, not a training or performance recommendation. No eta adaptation is selected or authorized here.

## Quantitative finite-T lower bound

Use any proved upper bound b<1 for c_star. One cheap choice is b=rho/sqrt(rho²+4 eta²); it can be tightened to min(b,sum|a_i|), since ||C-I||<=sum|a_i| also bounds the projective turning sine. Let c=|psi_s^H psi_o| and

g=max(0,c*sqrt(1-b²)-sqrt(1-c²)*b),
K0=1+sum|a_i|, K1=1+rho/eta,
f_j=E-e_j+q_j+i(eta+gamma/2), dmax=max(q_s/|f_s|,q_o/|f_o|).

The proposed lower bound is

T_so >= gamma²*q_s*q_o*g² / [|f_s*f_o|²*K0²*(1+(1+c)*K1*dmax)^4].

Proof outline. Put y=C^-1 psi_o. Its projective turning sine relative to psi_o is at most b (the inverse has the same maximal turning angle). The two-vector projection inequality gives |psi_s^H y|>=||y||*g>=g/K0. The inverse identity C^-1=I+X_c A_c^-1 X_c^H, with ||X_c||²<=rho and Im(A_c)=eta I, gives ||C^-1||<=K1. For U=[psi_s,psi_o], ||U||²=1+c and ||U^H C^-1 U||<=(1+c)K1. The two-terminal determinant factor D=det[I-(U^H C^-1 U)diag(q_s/f_s,q_o/f_o)] has |D|<=(1+(1+c)K1*dmax)². Substitute these estimates into the exact off-diagonal determinant identity and square the amplitude.

The lower bound is positive only above the angular cap, and may be far below the fixed epsilon1e-12. A bound below epsilon gives no practically useful log-score separation. A positive bound must still be compared with the actual numerical tolerance and scoring scale. This is not an empirical causal result, global class-margin guarantee or reason to change the running head.

## Bounded checks and next empirical question

mass-damping-fixture.py independently builds dense token Hamiltonians on synthetic free-state inputs. It checks the sharp zero witness, the near-parallel finite bound, the universal cap and dense T>=bound in bounded fixed fixtures. This checks algebraic consistency, not actual architecture attainability or data utility. The next real checkpoint question remains how many frozen examples have a nonvacuous angular/epsilon certificate; do not choose cohorts after observing performance.
