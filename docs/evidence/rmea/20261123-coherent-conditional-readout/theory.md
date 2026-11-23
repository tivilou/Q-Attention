# Prospective word-coherent conditional readout

Status: exploratory candidate, not accepted novelty/theory/L5. Ordinary scoped L4 remains unchanged.

## Task link and limits

On fixed complete-model reusedVALID32, four semantic-label-supported misses survive. NativeQ cue-uniform target-row relaxation crosses only1/4; one has all-word target-row upper margin−.6211585 with other values/bias/rivals fixed. The other complete models do not share that one-row obstruction. Attention sharpening alone is insufficient for that conditional case, not for all possible joint row or value changes. No shared training solution follows.

## Quantum construction

Valid words t=1..T, prepared unit states psi_t in C^8, relation unit effects phi_r, amplitude a_rt=phi_r†psi_t, signed word values u_rt, epsilon=1e-6 exactly as native. Define |Psi>=T^(-1/2) sum_t |t>|psi_t>, word dephasing D_W, and

rho_lambda=[(1−lambda)D_W(|Psi><Psi|)+lambda|Psi><Psi|+(epsilon/T)sum_t |t><t|tensor I_8]/(1+8epsilon).

Use postselection F_r=|+><+| tensor |phi_r><phi_r|, with |+>=T^(-1/2)sum_t|t>, and observable V_r=sum_t u_rt |t><t| tensor I_8. The real generalized weak conditional value is

z_r−b_r=Re tr(F_r V_r rho_lambda)/tr(F_r rho_lambda).

Set s_r=sum_t a_rt, v_r=sum_t a_rt u_rt, Q_r=sum_t|a_rt|². The explicit ratio is

[(1−lambda)sum_t|a_rt|²u_rt+lambda Re(s_r* v_r)+epsilon sum_t u_rt] / [(1−lambda)Q_r+lambda|s_r|²+epsilon T].

lambda=0 is exactly native Born+epsilon pooling. lambda=1 is the proposed word-coherent endpoint; no lambda search or learnable blend. Its coefficients beta_rt=[Re(s_r* a_rt)+epsilon]/[|s_r|²+epsilon T] sum to one, may be negative, and preserve constant word utilities. This differs from the closed shared-null differential, whose zero-sum coefficients discarded token-constant utility; no class-reference subtraction is used here.

With N_r=sum_t max(−beta_rt,0), any real utilities in [m,M] obey m−N_r(M−m) <= z_r−b_r <= M+N_r(M−m). This is the known Jordan decomposition of a signed measure, not new theory. A negative coefficient is necessary to escape the original convex hull, not sufficient for correct prediction. Postselection success probability is [(1−lambda)Q_r+lambda|s_r|²+epsilon T]/[T²(1+8epsilon)]. Small success probability trades against large weak values; epsilon alone gives finiteness, not a good conditioning or learning guarantee. No hardware/sample/query speedup claimed. Coherent preparation of the word register and controlled word-state circuit is an explicit ideal-model assumption, not an existing qRAM or hardware deployment.

Global phase of every relation effect cancels. Independent word-state phases do not cancel; the actual controlled preparation circuit fixes those relative phases. Native Born-only training never identifies them, so frozen native phases cannot fairly establish trained utility or kill this family. Fit all arms from a common random initialization; do not infer success from a label-aware frozen swap.

## Internal-dephasing and classical controls

The diagonal arm dephases only the8-dimensional internal register before postselection, retaining the word register. Let P_k=sum_t psi_tk. Its word coefficient numerator is sum_k |phi_rk|² Re(P_k* psi_tk)+epsilon, denominator sum_k |phi_rk|²|P_k|²+epsilon T. It may still have negative coefficients: it disables internal coherence, not word coherence. It is not the native dephased head or an all-coherence-disabled model.

The classical arm uses normalized real eight-dimensional linear keys/query and the same signed conditional formula, so it too permits negative weights. It shares values/decoder/bias, inputs and training order; its router is smaller, so do not assert exact total capacity matching or quantum necessity. The native arm is the unmodified Q-RMEA head. Quantum/diagonal/native have identical complete initial states and parameter count. A later formal study must additionally include strong common-protocol softmax/nonnegative controls; this bounded planning screen does not certify superiority over them.

## Primary source and theory boundary

Pusey, arXiv1409.1535v2, pages1–3, Eq1 and the projector decomposition establish weak-value definition and possible signed coefficients as prior art. PDFsha b364a95a8feaf7f609239d3bf63659dc29b76d8b30733f80dca976156c0a3c3b. Its contextuality theorem has further operational assumptions not tested here; signed pooling or a simulation anomaly alone does not invoke it. Weak values, dephasing and convex escape are known tools; the task-specific mechanism and novel theorem remain unestablished. Bounded arXiv title/abstract query found no attention-mechanism hit, which is not novelty clearance.

## One fixed prospective fit

Reuse the existing4192fit/4000assessment exact-sentence-disjoint TRAIN planning split, seed13, batch32, micro4, float64, AdamW.001/.01, clip1;512updates/arm and final-only assessment after all fits finish. No officialVALID/TEST in execution, no raw Windows data, no new encoder, no COS, no lifecycle-matrix rerun. Use coherent/diagonal/classical/native modes through the already accepted two-lane task-screen adapter with only the plugin-path and truthful budget text changed.

Question: can the coherent conditional readout learn correct-label repairs at no greater nullFP, while matching or exceeding native and signed-classical CE/Macro, rather than merely creating high-magnitude scores? Existing fixed screen gates apply. A failed exact512profile closes its promotion without changing seed, split, epsilon, lambda, budget or control. A short failure does not prove family impossibility. Numerical failure is reported as such, not converted into task evidence. No formal experiment or L5 promotion on this screen alone.
