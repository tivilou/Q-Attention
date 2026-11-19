# R99 v001: a fixed-spectrum query rotation and its finite-step comparison

Status: primary mathematical hypothesis, exploratory only. No implementation,
numerical execution, new task evidence, novelty acceptance or L5 promotion.
R96's joint-training point result is the empirical motivation. R95's negative
fixed-checkpoint classification result remains stopped. This is a different
operator path, not an eta/epsilon/seed/epoch rescue of that experiment.

## 1. Setting, provenance and the actual question

Use R97's fixed normalized states in dimension d=8 and fixed real utilities:

    P=phi phi*, Q=I-P, rho_t=psi_t psi_t*,
    S=sum_t rho_t, U=sum_t u_t rho_t, F=tr(P S)>0,
    E0_e=(P+e I)/(1+d e), D_e=tr(E0_e S),
    f0_e=tr(E0_e U)/D_e, A_e=U-f0_e S,
    N_e=sqrt(1+||A_e||_F^2), B_e=A_e/N_e.

The implemented R93-v002 map uses

    E^G_e(eta) = exp(log E0_e + eta B_e)/tr exp(...).

Token weights are tr(E rho_t)/tr(E S), with original u in final readout.
The source epsilon is fixed at 1e-6. Limits below are mathematical statements
for fixed states/utilities; they authorize no parameter sweep or retraining.

Question: does this log-exponential path become nearly fixed when E0 approaches
a pure projector, and can an explicitly different fixed-spectrum path move the
query without that suppression? Neither answer by itself establishes task gain.

Let f*=tr(PU)/F, A*=U-f*S, N*=sqrt(1+||A*||_F^2), and
C=||P A* Q||_F^2. Then tr(PA*)=0, A_e=A*+O(e), N_e=N*+O(e).

## 2. Finite-eta Gibbs suppression, including the deployed eta=1

**Proposition A (fixed bounded eta).** Put L_e=log((1+e)/e). For eta in any
fixed bounded interval [0,T], as e decreases to zero,

    E^G_e(eta) = P + (eta/L_e)(P B* Q + Q B* P) + O(L_e^-2),
    f^G_e(eta)-f0_e = 2 eta C/(L_e N* F) + O(L_e^-2).       (A)

Here B*=A*/N*. Constants depend on the fixed inputs, F and T. This is not
uniform over F approaching zero or over separately retrained representations.

Proof sketch with an explicit spectral-gap argument. Remove the scalar part
of log E0: the exponential is proportional to exp(L_e P+eta B_e).
The perturbation norm is at most T. For sufficiently large L_e, its leading
eigenvalue is simple and separated from all others by at least L_e-2T.
Writing its normalized leading vector as a phi+v with Qv=v, the Q-block
eigen-equation gives

    v = (eta/L_e) Q B* phi + O(L_e^-2), a=1+O(L_e^-2).

The leading spectral projector therefore has the first expansion in (A).
The total normalized exponential weight on the other d-1 eigenvectors is
O(exp(-L_e+2T))=O(e), absorbed in O(L_e^-2). Corrections B_e-B*=O(e) are
absorbed as well. The score quotient denominator tends to F>0. Since
tr(PA*)=0 and tr[(P B* Q+Q B* P)A*]=2C/N*, quotient expansion gives (A).

This extends the R97 eta-zero derivative comparison to a fixed finite eta;
it does not supply an error estimate at e=1e-6 or prove that our data occupy
the asymptotic regime. At fixed eta the Gibbs gain still vanishes in this limit.

## 3. A different path: rotate the projector and preserve its eigenvalues

Define an anti-Hermitian generator and effect

    K_e = [A_e,P]/N_e,
    V_e(eta)=exp(eta K_e),
    E^R_e(eta)=V_e(eta) E0_e V_e(eta)*.

V is unitary, so E^R has exactly E0's eigenvalues, trace and positive floor.
The original additive identity floor is unchanged. At eta=0,

    (E^R_e)' = (e_plus-e_minus)(P A_e Q+Q A_e P)/N_e,
    (f^R_e)'(0)=2(e_plus-e_minus)||P A_e Q||_F^2/(N_e D_e). (B)

For the actual R97 Gibbs cross derivative the coefficient is instead
ell=(e_plus-e_minus)/L_e. Thus the two cross derivatives differ by the factor
L_e. Larger local response alone is not attribution to a special geometry.
A future comparison must disclose or match this response scale.

**Proposition B (closed form and nonvanishing pure-limit own-score gain).**
Let w=Q A* phi=A*phi, s=||w||. If s>0, let v=w/s and t=s/N*. Then

    exp(K*)phi = cos(t) phi + sin(t) v =: q,
    lim_e->0 f^R_e(1) = q* U q/(q* S q),
    q* U q/(q* S q) > f*.                                (C)

If s=0 the limiting rotation is the identity on phi and no positive gain is
claimed. For s>0 this gives a fixed nonzero own-score gain, whereas (A) vanishes.

Proof. K*phi=w/N*, K*v=-(s/N*)phi; K* is zero on the orthogonal complement
of span(phi,v). These are a two-dimensional rotation, giving q. Since U and S
have range inside the span of token states, w=A*phi belongs to range(S).
If sin(t) is nonzero, w*q=s sin(t) is nonzero, so q is not in ker(S).
If sin(t)=0, q=+/-phi and F>0. Thus q*S q>0 in either case.

Write c=v*A*v, a real number. Then

    q* A* q = sin(t)[2s cos(t)+c sin(t)].

Here |c|<=||A*||_F<N*, and 0<t=s/N*<1. Consequently the bracket exceeds
N*[2t cos(t)-sin(t)]. Using cos(t)>=1-t^2/2 and sin(t)<=t shows
2t cos(t)-sin(t)>=t(1-t^2)>0. The quotient gain is positive. Continuity in e
gives (C). This proof applies to the pure-limit centered A*, not an asserted
universal finite-epsilon theorem at the deployed source epsilon.

The exponential need not be implemented as a general matrix exponential:
the generator has rank at most two. At any finite e, replace w by Q A_e phi
and use the same two-dimensional closed form with s_e=||w|| and N_e.
At s_e=0 take phi unchanged. This is an algebraic limit, not a clipping rule.

## 4. Why this is a lead, not a classification theorem or novelty claim

The unitary-orbit/double-commutator construction is established optimization
geometry. At the pure limit the sphere gradient of the Rayleigh quotient is
2 A*phi/F, so this query update is a scaled classical Riemannian gradient step.
Classical simulability is not a reason to reject ordinary task utility, but this
identity prevents claiming that the rotation itself is a new theorem or an
exclusive quantum resource. Brockett's 1991 double-bracket paper is a nearest
work lead; metadata identity alone is not original-paper reading or clearance.

Both (A) and (C) concern each class's own score. Consider two tokens
psi_+=(|0>+|1>)/sqrt(2), psi_-=(|0>-|1>)/sqrt(2), phi=|0>, utilities (+a,-a).
Then S=I on that plane, U=a sigma_x and f*=0. The rotated score is

    g(a)=a sin(2a/sqrt(1+2a^2)).

For 0<a<=1/2, g is strictly increasing: its sine argument increases and stays
below 1<pi/2. With two classes a_y=1/4 and a_k=1/2, equal initial logits and
biases, both scores strictly increase but g(a_k)>g(a_y). The true-class CE
increases from log(2) to log(1+exp(g(a_k)-g(a_y))). This embeds into d=8.
It preserves the R95 lesson and blocks an unconditional classification claim.

## 5. One prospective decision, before implementation

Retain only as an exploratory candidate: fixed-spectrum utility-conditioned
query rotation, with the same trained parameter count, source epsilon and
native final utility readout. It changes the operator path and does not add
a tunable coefficient. No full formal experiment is assigned here.

A future bounded joint-training comparison would need native, R96 Gibbs,
rotation and a disclosed response-matched classical/query-gradient control,
common initialization/data/order/budget and a frozen primary metric. Existing
repeated train-development data would remain discovery only. No extra seeds,
epochs or coefficient search may rescue a failed gate. That contract and its
implementation need separate exact review and shared scheduler admission.

First discriminating observations would be actual finite-step score/attention
movement, signed class competition and matched-control task metrics. Mere
nonzero cross activity or a faster 8x8 computation is insufficient. Substantive
near-neighbor novelty and appropriate independent evidence remain unproven.
Existing accepted scoped L4 and all stopped branches are unchanged. L5 NOT_ACCEPTED.
