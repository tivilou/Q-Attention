# Finite normalized-effect steering and a pure/mixed rank boundary

Status: primary-model derivation, awaiting independent check and numerical witness.
No task utility, natural-text reachability, novelty or L5 claim. The previous fixed2chord
construction and its outcome remain unchanged. This changes the feasible problem,
from raw mention powers plus denominator to the actual normalized mention weights.

## Exact normalized problem

For unit token states psi_i in C^d, fixed real utilities u_i, epsilon>0, T valid tokens,
let A_i=psi_i psi_i^H and Q_i=A_i+epsilon I. For any density effect rho>=0,Tr rho=1,
score_i=Tr(Q_i rho), B=sum_i Q_i>0, and alpha_i=Tr(Q_i rho)/Tr(B rho).
Original pure Q-RMEA is the special case rho=phi phi^H. All class effects remain
separate binary effects, not one joint POVM. Mixtures use Born scores before normalization.

Fix native normalized mention weights a_j. Holding all a_j automatically holds total
context attention; it does NOT require holding each raw score or the denominator.
The old raw+denominator constraint is a subset of this feasible set. Its closed
fixed-step construction is not being rerun with weaker thresholds.

Define Y=rho/Tr(B rho). Then Y>=0, Tr(BY)=1, and Tr(Q_jY)=a_j.
Conversely rho=Y/Tr Y is a density effect; Tr(BY)=1 implies its attention equals
Tr(Q_iY). Thus this is a bijection between normalized-effect attention and a compact
PSD affine slice. Equivalently Z=B^(1/2) Y B^(1/2) has trace1 and mention observables
D_j=B^(-1/2)Q_j B^(-1/2). The finite logit support function is the SDP

max Tr(CY), Y>=0, Tr(BY)=1, Tr(Q_jY)=a_j,
where C=sum_i u_i Q_i. A fixed bias adds a constant and does not change the optimizer.

Its dual value is inf lambda0+sum_j lambda_j a_j subject to
lambda0 B+sum_j lambda_j Q_j-C>=0. Weak duality always gives an upper certificate;
since B>0, a sufficiently large lambda0 gives dual strict feasibility. Primal
feasibility and compactness plus dual strict feasibility imply zero duality gap
and primal attainment. General dual attainment is not asserted; explicit witness
certificates below are attained and can be checked directly.

## Complex rank bound and a sharp existence threshold

An optimal extreme Y exists. If it has rank r and r^2>m+1 independent real equality
constraints, there is a nonzero Hermitian H supported on its range that annihilates
all equality traces. For small positive delta both Y+delta H and Y-delta H stay PSD
and feasible, contradicting extremality. Hence some optimum has r^2<=m+1
(or the smaller effective number of independent equalities). Congruence and positive
rescaling preserve rank. In particular, for m<=2 the mixed SDP optimum is attained
by a pure effect. This is a known complex SDP rank-reduction fact applied to this
normalization, not a new general rank theorem. It does not ensure reachability by
the source's12-angle circuit, nor a single global trained effect satisfying every example.

For m=3, rank2 can strictly improve on EVERY pure effect under the same exact
mention attention. The following finite witness makes the threshold sharp in the
existence sense, not for every frame/example.

## Explicit d=8 witness

Let q-space be span{e0,e3}, auxiliary e2. In q coordinates set
s_x=(1,1)/sqrt2, s_y=(1,i)/sqrt2, s_z=(1,0).
Let the three mention token states be psi_j=sqrt(2/3)s_j+sqrt(1/3)e2.
Complete each psi_j to an orthonormal basis of C^8; append one computational basis.
There are32 unit token states and sum_i A_i=4I. Let only the two appended context
tokens e0/e3 have utility1, all other tokens utility0. Native effect rho0=e2e2^H
and mixed rho*= (e0e0^H+e3e3^H)/2 both give every mention probability1/3.
Because B=(4+32epsilon)I, their normalized mention weights and context mass agree.
The mixed effect has logit (1+2epsilon)/(4+32epsilon), the native pure has
2epsilon/(4+32epsilon). The mixed optimum has q-mass1, hence is optimal.

For any feasible pure vector w, write its q part q, auxiliary amplitude t, remaining
mass o, with r=||q||^2 and r+|t|^2+o=1. The q Bloch vector b has norm r.
Each mention constraint gives
b_j=o-2sqrt2 Re(conj(t) s_j^H q).
Let lambda=(3+sqrt3)/2, the largest eigenvalue of sum_j s_j s_j^H. Then
r<=sqrt3 o+sqrt(8lambda r |t|^2).
For fixed r, set a=|t|^2 in[0,1-r]. The RHS is
sqrt3(1-r-a)+sqrt(8lambda r a). If r>3/(3+2lambda), it increases over that interval
and is at most sqrt(8lambda r(1-r)), giving r<=8lambda/(1+8lambda).
Otherwise the same upper bound holds since3/(3+2lambda)<=8lambda/(1+8lambda).
Thus every feasible pure logit is at most
(8lambda/(1+8lambda)+2epsilon)/(4+32epsilon), with a certified mixed/pure gap at
least 1/((1+8lambda)(4+32epsilon))>0. This is an upper bound, not claimed tight.

The native e2 and both mixture components (e0+e3)/sqrt2 and(e0-e3)/sqrt2 can be
prepared by the original source circuit with finite angles strictly inside(-pi,pi):
- mixture components: all angles0 except second-layer wire0 RY=+/-pi/2;
- native e2: first-layer all RY=pi/2; second-layer RY=(-pi/2,+pi/2,+pi/2); all RZ0.
First ring leaves |+++> fixed; the second local layer makes |011>, final ring maps
it to |010>. This correspondence must be tested against the actual pinned circuit.
The constructed token states and arbitrary orthonormal completions are STATE-PORT
fixtures: native token-preparation/natural-text reachability is not established.

## Mixture normalization and practical boundary

For rho=sum_l q_l phi_l phi_l^H, alpha(rho)=sum_l[q_l Z_l/sum_k q_k Z_k]alpha(phi_l).
Naively averaging normalized attentions with q_l is generally different. This also
exposes a multihead/classical convex-mixture collision; no quantum exclusivity claim.

The SDP is a per-example/per-relation oracle with frozen utilities. Its maximum
logit is not favorable CE/accuracy and cannot be used as a label-aware inference
rule. An optimal effect for one example need not be achievable by one globally
shared trained head. This feasibility/theory result alone admits no full training.
Nearest mixed-state attention, complex QCQP/rank reduction and quantum joint-numerical-
range originals must be checked before calling a method novel. A valid real-input
pilot would need a deployable shared plugin, matched rank/parameter/cost controls,
independent empirical evidence and a fixed task-level gate. Historical L4 is unchanged.
