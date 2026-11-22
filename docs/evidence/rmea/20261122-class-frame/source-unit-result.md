# Fixed class-frame source unit and its algebraic boundary

This is a development result, not a trained-model task result. The retained Nov20 class-frame lead was implemented at source revision95537f36b3c49354656de26ffd3ae2d0a5f64460. Exact Astra6.2/high mechanism+lifecycle review allowed only the CPU source-unit stage. The same reviewed bytes completed scheduler taskdc6a85240f7a413eaea316e6078fd70d / attemptbfc11f21917c4e2aaf462e8a854ba993 with exit0 and quiescent receipt. Zero optimizer steps and zero real-data reads were performed.

## Implemented map

For40 unit class queries phi_c in C^8 and unit token states psi_i, let P_c=phi_c phi_c^dagger, epsilon=10^-6, H=sum_c(P_c+epsilon I), W=H^(-1/2), E_c=W(P_c+epsilon I)W. This is the known square-root/PGM construction: E_c is positive and sum_c E_c=I. Per-class attention normalizes q_ci=psi_i^dagger E_c psi_i across valid tokens, then uses the unchanged source utilities and bias. The diagonal-frame control is positive but generally is not a POVM. Parameters/state_dict remain81028 and identical to the native head.

The inverse-square-root custom backward uses the continuous Hermitian divided difference -1/[sqrt(lambda_i)sqrt(lambda_j)(sqrt(lambda_i)+sqrt(lambda_j))]. The accepted scope is first derivatives, including repeated positive eigenvalues; higher derivatives are not certified.

## Executed findings

Repeated/distinct-eigenvalue complex gradcheck, dense effect and completeness calculations, masks, tight-frame/native equivalence, identical-query uniform limit, token phase gauge, native bitwise output parity, all-parameter finite gradients, one actual source-circuit effect directional derivative, and private fixture replay passed. Holding own query, token states and utilities fixed, changing competitor queries changed target attention in full and diagonal frames, while native attention was unchanged. This proves implemented parameter dependence, not better classification. Synthetic losses are unit outputs, not task metrics.

## Primary algebraic observation: static absorption

H depends on learned class queries only. For a frozen model H and E_c are global, independent of the sentence. At epsilon=0, additionally assume H is invertible. Put v_c=H^(-1/2)phi_c. Then q_ci=|v_c^dagger psi_i|^2. Its query norm cancels in per-class token normalization, so attention is ordinary Born attention using the normalized transformed query v_c/||v_c||. This is an exact fixed-state identity, not a proof that the original constrained12-angle state family can express every transformed query, nor that training dynamics or reachable hypothesis classes coincide.

For finite epsilon with W=(sum_c P_c+40epsilon I)^(-1/2), write a_ci=|v_c^dagger psi_i|^2 and g_i=epsilon psi_i^dagger H^(-1)psi_i. Then q_ci=a_ci+g_i. With A_c=sum_i a_ci and G=sum_i g_i over valid tokens, alpha_ci=(1-lambda_c) beta_ci+lambda_c gamma_i, where lambda_c=G/(A_c+G), beta_ci=a_ci/A_c, gamma_i=g_i/G. Epsilon>0 makes G>0; if A_c=0 take lambda_c=1 and arbitrary beta_c. Thus the floor contributes a common anisotropic background with class-dependent mixing. This exact decomposition does not bound its size for an ill-conditioned frame.

These are primary derivations and have no separate independent mathematical acceptance. The source-unit review predates this paragraph. No literature-priority or novelty claim is made. Competitor dependence by itself is insufficient evidence of contextual competition: a static frame can exhibit it.

## Disposition and next test

Keep this implementation as a reusable, first-order, source-tested8x8 primitive and an explicit static control. No full training is launched solely to show the known PGM construction or the above dependence. A next input-adaptive measurement must demonstrate a sentence-dependent effect under fixed learned parameters and distinguish it from global reparameterization and a common background; task utility still needs matched-budget empirical testing.

The private744838-byte synthetic fixture remains on the3060 worker and replays with maximum error0. It is rebuildable from source and is not sample-trace.v2 or a Case Study; no model/cache/checkpoint was downloaded to Windows or uploaded to COS. The formal plugin/trace readiness remains pending. The old scoped L4 is unchanged; L5 NOT_ACCEPTED.
