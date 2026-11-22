# One native effect, natural decisions, and biseparable replacements

Prospective exploratory hypothesis: on a fixed, balanced set of native-correct VALID decisions, every trace-one biseparable replacement of one native relation effect reverses at least 1% by balanced weight. All token encodings, values, biases and rival effects remain fixed. The replacement family includes mixed effects and mixtures across cuts; it is larger than the original relation circuit family.

For example i define the frozen largest rival logit T_i, sign s_i=+1 for a correct target-relation win and −1 for a correct rival win. For any trace-one positive E, the signed target/rival margin has positive denominator D_i(E)=Σ_valid tr[E(ρ_it+εI)] and numerator tr(E H_i), where

H_i=s_i Σ_valid (u_irt+b_r−T_i)(ρ_it+εI).

Thus the nonlinear normalized readout yields linear decision inequalities in a common E. Let A_i=H_i/c_i with c_i=||H_i||op(1+10⁻¹⁰)+10⁻¹², weights w_i>0 summing to one, and 0≤λ_i≤w_i. Then W=−Σ_i λ_i A_i satisfies tr(EW)≤Σ_i w_i 1{tr(E A_i)<0} for every trace-one positive E, since ||A_i||op≤1. If, for every one-wire cut k, W−δI=P_k+PT_k(Q_k), P_k,Q_k≥0, every PPT mixture, hence every biseparable E, has tr(EW)≥δ. This lower-bounds strictly reversed decisions by δ. A failed sufficient certificate does not prove biseparable feasibility.

PPT mixtures and fully decomposable witnesses are established tools: Jungnitsch, Moroder and Gühne, *Taming multiparticle entanglement*, arXiv:1010.6049v2, main-text Eqs.1–4 (original source read). The source-specific candidate is a bound on actual normalized decoder decisions, not the separability relaxation itself. General priority remains unresolved; asymmetric fidelity attention can support related constructions.

Selection is frozen before inference: choose the most frequent positive TRAIN relation (tie: existing class index); in stored VALID order take first128 strict native-correct wins for that relation and first128 strict native-correct other-label wins where this relation is uniquely second. Use all available up to caps, with half weight per group. Fewer than32 in either group stops as under-supported, without another relation or subset. No TEST access, fitting, seed/epoch/checkpoint/threshold rescue. One native float64 inference pass, at most19584 examples, plus one SCS solve. Primary numerical gate: checked δ≥.01.

Native-margin replay, mask handling and MSB-wire partial transpose are the only changed-mechanism checks. The verifier clips λ, projects Q positive, reconstructs W, and lowers δ by 10⁻⁷(1+||W||op+max||Q_k||op). Report all residuals. This is a float64 stored-coefficient numerical certificate, not exact interval certification of the original real-valued circuit. Exact arithmetic certification and broader claims require a separately justified extension; they are not silently inferred from a solver success status.

The immutable seed13 native checkpoint is the same architecture as historical Q-RMEA, but it does not inherit the original fresh5 L4 evidence. Even a pass is conditional on selected exposed VALID decisions, not net Macro loss, mediation of the five-seed gain or independent confirmation. The statement is relative to the original three-wire tensor encoding: a general simultaneous basis change can make an individual relation vector a product vector without changing fidelities. Unrestricted classical complex-vector simulation remains equivalent. L5 is NOT_ACCEPTED.

The current failed relative-return three-arm run must finish unchanged and release its allocation before this new managed inference task starts. No changes to rmea.py, the trainer, workers, checkpoint/COS code, or their accepted lifecycle tests.
