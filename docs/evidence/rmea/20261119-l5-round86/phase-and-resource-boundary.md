# Coherent readout: phase quotient, erasure and resource boundary

Status: ANALYTICAL_DRAFT_FOR_EXACT_REVIEW. The separate assessment records the
completed review without changing these frozen bytes. No new method is selected.
This is a transfer analysis of a specified weak-value readout, not a novelty,
task-utility, quantum-advantage or L5 claim. No scientific runtime was executed.

## Scope and fixed definitions

For one fixed example and one relation, take n normalized token states psi_t,
a normalized query phi and finite real downstream utilities u_t. Set
c_t=<phi|psi_t>, D=sum_t c_t, A=sum_t u_t c_t. The unselected coherent model is

    |Psi> = (1/sqrt(n)) sum_t |t>|psi_t>,
    |chi> = |+_n>|phi>, O=sum_t u_t |t><t| tensor I,
    W=A/D, f=Re(W), p=|D|^2/n^2,  D != 0.

Utilities, queries and token projectors are held fixed under the phase changes
below. This is a prospective different readout; it is not the executed native
RMEA head. In particular this coherent pure-state exploration has no EPS branch.
The exact EPS-including flagged-mixture equivalence is in the frozen R85 record.
No old L4 support or arbitrary phase reachability by native trained parameters
is asserted. Relative phases in a controlled joint preparation are physical;
independent global phases of separate projectors are not a joint-state gauge.

## 1. Exact classification of phase-independent real readout

Let S={t: |c_t|>0}, k=|S|>=1, and write c_t=a_t exp(i theta_t), a_t>0 on S.
Ask whether f is the same real C for **all** independent theta_t with D != 0.
This is the precise requirement for this readout to descend to the interface
containing only separate token projectors, u and the query projector.

Classification:

- k=1: f=u_t.
- k=2: f is phase-independent iff either u_1=u_2, or a_1=a_2.
  In the latter case C=(u_1+u_2)/2, even when u_1!=u_2. W itself can retain an
  imaginary part, and the destructive-interference zero remains excluded.
- k>=3: f is phase-independent iff all active utilities equal C.

Proof. On D!=0, f=C is equivalent to Re(A D*)-C |D|^2=0. Expanding gives

    sum_i (u_i-C)a_i^2
      + sum_{i<j} (u_i+u_j-2C) a_i a_j cos(theta_i-theta_j) = 0.

The nonzero-D subset is dense in the phase torus: the nonzero analytic
trigonometric polynomial D cannot vanish on an open set. Continuity therefore
extends the displayed identity to the whole torus. Uniqueness of its distinct
Fourier coefficients yields u_i+u_j=2C for every active pair, and
sum_i(u_i-C)a_i^2=0. Three active indices force all u_i=C. For two indices the
remaining constant coefficient is (u_1-u_2)(a_1^2-a_2^2)/2. These conditions are
also sufficient, proving all cases. Tokens with zero overlap do not contribute
to A or D; the statement concerns one fixed relation, not simultaneous masks
or all model outputs.

This is a narrowly derived transfer lemma with no originality claim. It does
not prohibit explicit controlled phase encodings. Such encodings specify a
different architecture/resource interface and require a task-level premise.
Canonical phase alignment c_t -> |c_t| gives the ordinary nonnegative pooling
sum u_t sqrt(p_t)/sum sqrt(p_t), p_t=|c_t|^2. That **specified** gauge-fixing rule
is a square-root score transform, not new phase information or a theorem about
every possible phase-selection rule. Its denominator excludes all-zero overlaps.

## 2. What independent phase erasure actually recovers

Let independent theta_t be uniform on [0,2pi) on every preparation. Define
p_theta=|D_theta|^2/n^2 and W_theta=A_theta/D_theta on successful phases.
Extend p_theta W_theta continuously as A_theta D_theta*/n^2 at D_theta=0.
Assume sum_t |c_t|^2>0. Orthogonality of phase characters gives exactly

    E[p_theta] = sum_t |c_t|^2/n^2,
    E[p_theta W_theta] = sum_t u_t |c_t|^2/n^2,
    E[p_theta W_theta] / E[p_theta]
      = sum_t u_t p_t / sum_t p_t.

Thus **success-weighted** phase erasure returns unsmoothed Born pooling. The
same holds for its real part. This is not E[W_theta], not the expectation of an
unweighted per-example ratio, and not an exact finite-sample ratio identity.
It is also not the EPS-smoothed native formula. With fixed u, the physically
phase-erased joint mixed state gives the same known weak conditional ratio.
No statistical classifier performance conclusion follows from this identity.

## 3. Exact amplification/success bound, and its applicability

Let m=<Psi|O|Psi>=sum_t u_t/n and
sigma^2=<Psi|(O-m)^2|Psi>=sum_t (u_t-m)^2/n. For any normalized postselection,

    p |W-m|^2 <= (1-p) sigma^2.

Proof. v=(O-m)|Psi> is orthogonal to |Psi> and ||v||^2=sigma^2. Only the
component of |chi> orthogonal to |Psi> contributes to <chi|v>; it has squared
norm 1-p. Cauchy-Schwarz gives |<chi|v>|^2 <= (1-p)sigma^2, while
<chi|v>=<chi|Psi>(W-m). This includes p=1 and sigma=0 whenever W is defined.
For a real excursion |Re(W)-m|>=L>0 it implies

    p <= sigma^2/(sigma^2+L^2).

This is a known elementary weak-value geometry bound, not new metrology theory.
For physical repeated preparation and postselection, success probability must
enter the resource contract. A weak-meter implementation also needs a valid
finite-coupling/noise/estimator analysis; W alone does not describe that device.
For a classical simulator directly evaluating A/D, p is **not** a measured
runtime or shot cost and this bound does not prove bad NLP performance.
Near-zero D instead calls for explicit numeric stability and matched functional
controls. Neither a larger logit nor this bound by itself selects/rejects a
task-defined coherent model.

## Original-source boundary and research consequence

Farinholt/Ghazarians/Troupe, arXiv:1512.02113v2, p3 Eq1-2 (R85 reading), gives
the known weak-value ratio. Ferrie/Combes, arXiv:1307.4016v3, read pp1-3 and
visually inspected p3, studies a weak-coupling Gaussian-meter estimation and
detection setup; p3 Eq18-22 compares specified estimators while retaining the
number of successful outcomes. Its metrology conclusion is not a theorem about
our deterministic relation classifier. Later expansions, detection proofs,
supplement and implementation are not audited here. These sources establish
nearby known primitives and the need for a resource contract, not priority
clearance for our lemma or a performance verdict for all coherent architectures.

The conjunction closes only the proposed shortcut: unchanged separate-projector
access + phase-free anomalous amplification + inherited old L4. It does not close
explicit phase models. No task-defined phase mechanism is currently selected,
so no coherent training, phase grid, parameter rescue or formal handoff follows.
Next discovery should supply an independently motivated task/information link
or investigate a structurally different quantum readout; a renamed ratio is
insufficient. R85 independent ideation NONE, scoped L4 and stopped families stay
unchanged. L5 NOT_ACCEPTED.
