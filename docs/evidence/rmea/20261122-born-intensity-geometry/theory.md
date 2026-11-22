# Born intensity preservation: an exploratory geometry lead

Status: exploratory mathematics, not a selected implemented method, utility or L5.
Current protected-context full seed13 continues unchanged on taskbee992dc.
Historical scoped L4 concerns the earlier Q-RMEA and is not inherited here.

## Observed code and open question

Executed source1c2f39e0, `rmea_protected_context.protected_projector` spans every
masked mention state in C^8. The correction has P delta=0. Thus it preserves
each complex mention amplitude, and hence its squared modulus. If rank(P)=8,
delta=0. This is known linear constrained filtering, not a new result. The
1e-10 truncation means numerical full rank and exact span are distinct.

The downstream attention uses only |w^H psi_i|^2+EPS. Does exact preservation
of those intensities and unit norm leave context-sensitive motion even when
the mention states span the whole Hilbert space? This is a different feasible
set, motivated by phase-retrieval fibers. It does not alter the running run.

## Local statement and proof

Fix mention states psi_1..psi_m and a unit vector phi in C^d. Treat C^d as
R^(2d), with real inner product Re(x^H y), and define

F(w)=(||w||^2, |psi_1^H w|^2, ..., |psi_m^H w|^2).

Assume DF has constant real rank r in a neighborhood of phi. The local fiber
F(w)=F(phi) is a smooth manifold of real dimension2d-r. Its tangent is

K={v: Re(phi^H v)=0 and
       Re(conj(psi_j^H phi) psi_j^H v)=0 for everyj}.

This follows directly by differentiating the quadratic constraints and applying
the constant-rank theorem. The nonzero global-phase direction i phi is always
in K. Locally quotienting the free S^1 action leaves dimension2d-r-1.
For independent norm+mention constraints, r=m+1, so dimension=2d-m-2.
In d=8 this is positive when m<=13. This is conditional on regularity, not an
assertion about every frame, phi, trained circuit or dataset example. With an
orthonormal mention basis the norm constraint is redundant; r=d rather than
d+1 at nonzero coordinates, and the quotient dimension is d-1.

Let C be PSD Hermitian, lambda>0, E(w)=w^H Cw+lambda||w-phi||^2. At phi the
real Euclidean gradient is g=2Cphi. If t=-Proj_K(g) is nonzero, smoothness of
the fiber gives a feasible curve w(s) with w(0)=phi,w'(0)=t. Its derivative is
E'(0)=<g,t>=-||Proj_K(g)||^2<0. Therefore there is exact, norm/intensity-
preserving local descent. The phase direction alone cannot create this because
<g,i phi>=0. This is standard constrained geometry, not a novelty certificate.

By comparison, preserving all complex amplitudes requires Psi^H(w-phi)=0.
If Psi has complex rankd, its only solution is w=phi. Thus full span alone
does not eliminate intensity-preserving motion, although it eliminates the
current amplitude-preserving correction.

## Exact full-span witness in dimension8

Take the eight mention states e_j, phi=(1,...,1)/sqrt8,
u=(e_1+i e_2)/sqrt2, C=u u^H, and
w(t)=(exp(it),1,...,1)/sqrt8.

Every mention probability remains1/8 and ||w||=1, while

|w(t)^H u|^2=(1-sin(t))/8,
E(w(t))-E(phi)=-sin(t)/8+lambda(1-cos(t))/4.

The derivative at0 is-1/8 for every finite lambda. For the current numerical
ridge lambda=1/8 and t=pi/2, context power falls from1/8 to0 and energy falls
by3/32. The amplitude-protected full-span model stays exactly phi. Adding the
actual EPS=1e-6 preserves mention raw scores exactly; normalized mention weights
still change through their common denominator. This is a state-port witness,
not proof that the trained circuit/real text realizes these eight states.

## Limits and falsifiers

- Pointwise rank alone is insufficient at a singular level. With one mention
  psi_1=phi and ||w||=1, preserving probability1 forces w=e^(it)phi even when
  d>1 and m=1. The Jacobian is singular and the regular-fiber claim does not apply.
- A frame that identifies phi up to global phase leaves no useful motion.
  Global phase retrieval thresholds must not be substituted for local fiber
  dimensions; the original theorem's generic/all-signals quantifiers matter.
- Changing attention can leave logits identical when all token utilities are
  equal, or otherwise lie in a decoder nullspace. Lower context energy can
  hurt useful context. No universal accuracy, margin or CE improvement follows.
- Unit norm bounds an individual rank-one effect on unit token states. It does
  not make the collection of class effects a normalized POVM.
- A general feasible vector can be reached by some abstract unitary from phi;
  no efficient implementation by the present shallow circuit is established.
- Exact nonlinear constraints require an actual stable differentiable update;
  a tangent step without retraction preserves intensities only to first order.
  No new fitting implementation is admitted by this note.

## Next discriminating step

First independently check the witness, derivative and regularity boundaries,
then consult the exact mathematics. Read the nearest phase-retrieval originals
before asserting any substantive novelty. If the route survives, freeze one
small no-fit test of exact versus tangent-only preservation and task-visible
motion on a predeclared TRAIN cohort through the existing scheduler after the
current run. Do not change its controls or add a rescue experiment mid-run.
The actual applicability/solver remain open. Existing L4 is preserved unchanged.

## Reviewed local algorithm, not a universal solver

Astra6.2/high independently checked the statements and witness. It closes the
universal claim that arbitrary mention sets always allow useful smooth motion;
the certainty and phase-identifying cases are genuine obstructions. Its
further construction applies only to full-row-rank J=DF(phi).

Let B be any right inverse of J, choose a unit t in kerJ and
D=(1+sum_j||psi_j||^4)^(1/2). Let
q(x)=(||x||^2,(|psi_j^H x|^2)_j), in real coordinates. Set
s=min(.01,1/(8||B||D)) and iterate n=-Bq(st+n) in rangeB starting at0.
For ||n||<=s, ||st+n||<=2s, so the image norm is<=4||B||Ds^2<=s/2.
The Lipschitz constant is<=4||B||Ds<=1/2. Banach's theorem yields a unique
fixed point in that ball, and
F(phi+st+n)-F(phi)=Jn+q(st+n)=0 exactly. Finite floating-point iteration
only gives a measured residual. Using ||B||_F in place of ||B|| is a valid
conservative bound. Inputs and step/direction branches must stay regular for
implicit differentiation; no global or rank-crossing smoothness is claimed.

For a fixed class let L=sum_i alpha_i u_i+bias, including EPS in alpha and
masking invalid tokens. Its derivative is
dL[v]=sum_i (u_i-(L-bias)) dp_i[v]/Z.
Let a=Proj_K gradE and c=Proj_K gradL. A feasible direction with dE<0 and
dL!=0 exists iff both a,c are nonzero, on this regular fiber. Necessity is
immediate; sufficiency follows because an open half-space cannot lie in the
kernel of a nonzero linear functional. An explicit direction is proportional
to -a/||a||-sign(<a/||a||,c/||c||>) c/||c||, choosing sign(0)=+1. It lowers
energy and changes that class logit; its logit sign is not a task utility claim.
The ridge objective is not generally phase-invariant, so the dimension quotient
describes feasible freedom; any quotient optimization would need a phase gauge.

This construction and its connection to task readout are still standard local
constraint calculus. The research lead is whether actual learned states admit
a numerically resolvable finite motion with a useful mechanism. The synthetic
test cannot settle that or certify substantive novelty.
