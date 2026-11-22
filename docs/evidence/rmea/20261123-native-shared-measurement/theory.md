# Shared native local-Pauli measurements: a planning qualification

The original source prepares a token state psi_t independent of relation r, then
uses binary effects P_r=|phi_r><phi_r|. No head or score is changed. Select t
uniformly, one of 27 three-qubit Pauli bases uniformly, then one of eight outcomes
with its Born probability. The known local-shadow snapshot is the tensor product
of (3|b_j><b_j|-I). Its expectation on psi_t is rho_t. Each physical copy therefore
provides an unbiased readout h_r=tr(P_r snapshot) for every relation; h_r can be
negative and is not itself a physical probability.

With original a_rt=b_r+u_rt-s and D_r=mean(p_rt+EPS)>0, C_r=D_r(z_r-s).
For X_r=a_rt h_r, E X_r=mean_t a_rt p_rt, and adding EPS*mean_t a_rt gives C_r.
The exact variance is mean_t a_rt^2 E[h_r^2|t]-(mean_t a_rt p_rt)^2.
All 27*8 outcomes are enumerated, including zero-mass endpoints in the conservative
support. Coefficient/readout extrema give the exact Cartesian support envelope.

For a large TOTAL state-preparation budget N, a shared estimator has marginal
standardized variance V_shared,r/(N C_r^2). A direct class-specific estimator with
N_r copies has V_direct,r/(N_r C_r^2). The real-valued oracle allocation equalizing
these quantities gives min max_r V_direct,r/(N_r C_r^2) = sum_r V_direct,r/C_r^2 / N,
when all C_r are nonzero and the usual zero-variance limiting cases are respected.
Compare H_shared=max_r V_shared,r/C_r^2 to that sum. This is a known allocation
identity and a population second-moment proxy, not actual stopping cost or a new
theorem. It omits bounded-range terms, integer allocations, simultaneous error
control and cross-class correlations. Zero contrasts remain explicit failures.

Threshold s is the previous oracle midpoint of exact native top-two logits,
planning only, not optimal or deployable. The first128 stored TRAIN rows are
development inputs, not independent evidence; no labels, VALID or TEST.
Success requires both frozen proxy gates in design.json, against direct |a|-token
sampling. Failure closes this exact uniform local-Pauli profile without rescue.
Success admits further design only, never a VALID study or L5 on its own.

Prior tools: Huang/Kueng/Preskill arXiv:2002.08953v2 p13 EqS17 supplies the snapshot.
Ma/Liu arXiv:2509.06218v3 pp2-3,6-8 stresses measurement reuse need not improve
full-stack cost. Their random sparse-matrix assumptions are not imposed on this
dense trained rank-one head; enumerate the actual matrices instead. No new
shadow theorem, novel quantum attention, sampled advantage or hardware speedup
is claimed. Classical feature, preparation, gate and readout costs remain open.

Only affected tests: inverse-channel reconstruction on three fixed complex states,
two effect means, nonconstant/zero-coefficient weighted outcome variance, and actual
native forward/probability/previous-contrast and direct-variance parity. Stable
runtime and checkpoint/COS tests are reused, not repeated. Private small arrays
remain on the GPU server's disk/host memory; no Windows payload or COS write.
