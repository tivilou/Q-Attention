# Selective dephasing and pointwise decision attainability

This is a new diagnostic of the ORIGINAL frozen model, not a rescue of the
closed outer-dephasing utility recipe. No new architecture, optimization or
quantum-necessity claim. The generalized Rayleigh tools below are standard;
source-specific empirical explanation and substantive novelty remain questions.

For each valid token let rho_t be its original rank-one density or the existing
selectively dephased density (all eligible nonentity context, entities protected).
Keep original signed utilities u_rt and bias b_r. Set
B=sum_t rho_t + EPS*T*I,
A_r=sum_t u_rt*rho_t + EPS*sum_t u_rt*I.
For ANY unit rank-one relation effect v_r, its logit is
b_r+(v_r^* A_r v_r)/(v_r^* B v_r).
B is positive definite because EPS>0. Its exact endpoint range is the extremal
generalized eigenvalues of (A_r,B), plus b_r. The forty effects are independent
binary measurements, not one joint POVM. Thus the supremum gold margin with
input-specific unrestricted effects is M=H_y-max_{r!=y} L_r, attained by choosing
the gold maximizing and every rival minimizing vector independently. M<0 rules
out a strict correct decision even in this enlarged family; M>=0 does not
establish circuit realizability, shared learned effects, or task utility.

Uniformly dephasing ALL tokens cannot enlarge any scalar logit endpoint range:
the dephasing dual maps an effect to a positive mixed effect; its quotient is a
positive-denominator weighted average of original pure-effect quotients. The
selective channel need not obey this inclusion, since protected and changed
tokens do not share the same channel. Two-token counterexamples in an embedded
qubit give expansion and contraction. For protected |+>, changed |+>, utilities
(+1,-1), the original gold quotient is identically zero; after dephasing only
the changed token its maximum is 1/(3+4EPS). A constant rival at .2 is initially
unbeatable and then beatable. Replace the changed state by |-> and the rival by
.5: the original gold maximum is 1/(1+2EPS), but the changed maximum drops to
1/(3+4EPS), reversing the boundary. Other dimensions have zero gold numerator.
Basis-diagonal states are a counterexample to any universal boundary change.

Frozen scientific hypothesis: BOTH directions occur among first8192 existing
TRAIN rows using the already qualified input-only masks and all40 output
classes. The finite prefix can lack some gold labels: no row is added/dropped,
missing gold classes are explicitly reported, all rivals remain in each bound.
Damage witness: native correct/context wrong AND M_context<0.
Repair witness: native wrong/context correct AND M_native<0.
Compute eigenproblems only on ALL correctness flips, never select a relation,
confidence, length or favorable case. Ineligible rows remain identity in both
modes. A sign is counted only below minus1e-7 AND a conservative numerical
uncertainty allowance. These are floating-point resolved obstructions, NOT
outward-rounded interval certificates. Independent SciPy generalized eigh
checks endpoints/margins/counts. Zero witnesses in EITHER direction closes this
exact two-direction hypothesis; no alternate mask/cohort/checkpoint/tolerance
rescue. Both positive establishes existence on exposed TRAIN only; report
fractions and allclass counts, not population significance/independent evidence.

Two allocated GPUs shard rows and reuse the accepted host runner. Zero updates,
no VALID/TEST, no COS/checkpoint, no restart. All raw moments/predictions stay on
GPU host. Compact report includes fixtures, both denominators/counts, numerical
uncertainty and fullclass counts. Original L4 retained; L5 not accepted.
