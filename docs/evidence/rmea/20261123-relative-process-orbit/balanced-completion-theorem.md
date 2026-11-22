# Positive-fidelity completion collision and conditional task utility

This note derives a fixed-encoding distinction for the relative-return mechanism actually under study. It is a mathematical construction inside the declared encoder and readout, not a result on natural text, an outcome of the running full study, or L5 acceptance. No experiment, selection rule or frozen TRAIN64 diagnostic changes.

## Circuit and scores

Use the original two-layer, three-wire circuit. Each layer applies RY then RZ on each wire, followed by the ordered CNOT ring 01,12,20. Let F denote that ring and write U=F L2 F L1. Physical angles are pi*tanh(raw), with wire0 the most significant bit. For relation V and token U, put W=V†U and d=8. The scores considered here are

\[
p_1=|W_{00}|^2,\quad p_S=|W_{00}|^4,\quad
p_D=\sum_j|W_{0j}W_{j0}|^2,\quad p_C=|(W^2)_{00}|^2.
\]

Also consider the known Choi-process fidelity F_e(W)=|Tr W|²/d² and the PSD envelope H/d²=F_e(W)p_1 from the accompanying [process-feature note](twisted-kernel-note.md). Equality for this particular process comparator is stronger than equality of prepared-state fidelity; it is not equality of every process observable.

## A collision at strictly positive fidelities

For every 0<phi<pi/2, let psi=pi-phi,

\[
A=R_Z(\phi)R_Y(\pi/2),\qquad B=R_Z(\psi),\qquad
W_{lo}=A\otimes B\otimes I,\qquad W_{hi}\simeq W_{lo}Z_0Z_1.
\]

Here ≃ ignores a global phase. These relative unitaries are realizable by the original twelve-angle encoder with finite raw angles. Choose identical second layers for V and both tokens, and choose the first-layer physical angles as follows; all unspecified angles are zero:

| Circuit | wire0 (Y,Z) | wire1 (Y,Z) |
|---|---|---|
| V | (0,-phi/2) | (0,-psi/2) |
| U_lo | (pi/2,phi/2) | (0,psi/2) |
| U_hi | (-pi/2,phi/2-pi) | (0,psi/2-pi) |

The common second layers and the two first-layer output rings cancel in V†U, leaving (L1_V)†L1_U. The single-wire identity

\[
R_Z(z-\pi)R_Y(-y)=iR_Z(z)R_Y(y)Z
\]

then gives U_hi≃U_lo Z0Z1. Since Z0Z1 fixes |000>, the two tokens prepare exactly the same density operator. Every displayed physical angle lies strictly inside (-pi,pi), so atanh(angle/pi) is finite. This construction need not be a regular point of the prepared-state map; positive fidelity does not establish generic differential rank.

For A, the two nonzero return paths have amplitudes

\[
A_{00}^2=e^{-i\phi}/2,\qquad A_{01}A_{10}=-1/2.
\]

Right multiplication by Z flips the second path's sign. B is diagonal and contributes only an irrelevant return phase. Consequently both tokens have

\[
p_1=1/2,\quad p_S=1/4,\quad p_D=1/2,
\]

while

\[
p_C(U_{lo})=(1-\cos\phi)/2,\qquad
p_C(U_{hi})=(1+\cos\phi)/2.
\]

The second wire balances the process trace: |Tr A|=sqrt(2)cos(phi/2), |Tr AZ|=sqrt(2)sin(phi/2), whereas |Tr B|=2sin(phi/2), |Tr BZ|=2cos(phi/2). Thus

\[
F_e(W_{lo})=F_e(W_{hi})=\sin^2\phi/8>0,\qquad
H(W_{lo})/64=H(W_{hi})/64=\sin^2\phi/16>0.
\]

At phi=pi/3, the coherent scores are 1/4 and 3/4, the process fidelities are 3/32, and both normalized envelopes are 3/64. This proves an exact collision for the specified fixed-encoding comparators, including a positive process-aware PSD comparator. An independently trained comparator may change its encoding or readout; this is not a learning-class separation.

## Compatibility with the actual shared token mapper

The native source uses one Linear(1028,12) preparation map on a 1024-dimensional feature plus four role indicators, followed by pi*tanh. It uses a Linear(1024,64) value map, a 40-by-64 decoder and forty biases. Their source identities are rmea.py SHA256 9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0 and frozen_feature_head.py SHA256 f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9.

The four synthetic feature ports h_lo,±=e0±e2 and h_hi,±=e1±e2, with any one fixed valid role indicator, realize this construction through one shared mapper. Set preparation columns0 and1 to atanh(vec(angle_lo)/pi) and atanh(vec(angle_hi)/pi), with all other columns and the bias zero. The e2 coordinate can therefore carry the value sign without changing preparation. Set the value map's first coordinate to (v/2)h_2, all others zero; choose opposite first-coordinate decoder rows for two active classes. Relation-angle parameters are atanh(vec(angle_V)/pi), duplicated for those two classes. Thus the readout construction below is compatible with the source shapes and shared parameterization.

These ports are synthetic. Reachability from actual frozen RoBERTa text, discoverability by optimization and trained task benefit remain empirical questions. The proof does not assume those questions answered.

## Exact utility criterion, including the forty-class readout

Consider a two-token conditional task using the two completions. For B in {-1,+1}, the high-score token has active-class value Bv/2 and the low-score token -Bv/2, with v>0. The second active decoder row negates these values. Both rows use the same relation V and zero bias. Let B be equiprobable, and let label Y in {-1,+1} agree with B with probability q. Labels on this constructed task occupy only the two active classes.

The remaining C-2 decoder rows are zero, with common bias c. This is a valid conditional construction for C=40; it is not a claim that the actual forty-class dataset has this label distribution. For C=2 those rows are absent. With EPS=epsilon=10^-6, native attention normalizes p+epsilon. Each equal-score comparator gives zero active logits. Coherent attention gives opposite active logits ±Bm/2, where

\[
m=\frac{v\cos\phi}{1+2\epsilon}>0,\qquad M=(C-2)e^c.
\]

The exact expected cross-entropies are

\[
L_{equal}=\log(2+M),\qquad
L_C=\log(2\cosh(m/2)+M)-(q-1/2)m.
\]

Therefore coherent attention improves this task **if and only if**

\[
(q-1/2)m>\log\frac{2\cosh(m/2)+M}{2+M}.
\]

For the binary case M=0, the right side is log cosh(m/2). The proof simply applies the actual EPS-normalized weight difference, (p_hi-p_lo)/(p_hi+p_lo+2epsilon), to the opposite values, then takes the softmax log partition and averages the signed-label term. The log-loss algebra is standard; the circuit collision and its propagation through this specific shared head are the source-specific statement.

## Orientation benefit is distinct from utility

Apply each of the eight common token branches U_t→U_t Z^s, keeping relations, values, masks, biases and EPS fixed. Because the second and third relative factors are diagonal, only the first branch bit changes these coherent score pairs. Four branches swap high and low, and four preserve them. Among seven nonoriginal branches, four flip the active logits and three preserve them. The partition function is unchanged, even with the other C-2 fixed logits, giving

\[
\Delta=\frac17\sum_{s\ne0}L_s-L_0=\frac47(2q-1)m.
\]

In this family, Delta>0 requires only q>1/2. It is weaker than the sharp utility condition above. At q=0.6 the following directly evaluated values illustrate the distinction:

| C | other bias c | m | L_equal | L_C | L_equal-L_C | Delta |
|---|---|---|---|---|---|---|
| 2 | absent | 0.5 | 0.693147181 | 0.674076984 | +0.019070196 | +0.057142857 |
| 2 | absent | 4.0 | 0.693147181 | 1.618149928 | -0.925002747 | +0.457142857 |
| 40 | -4 | 0.5 | 0.991767071 | 0.964803225 | +0.026963845 | +0.057142857 |
| 40 | -4 | 4.0 | 0.991767071 | 1.706617125 | -0.714850055 | +0.457142857 |

Thus a beneficial orientation can coexist with harmful confidence. The factor4/7 and the sharp threshold are exact for this constructed family, not an asserted equation for an arbitrary trained multiclass checkpoint. The already frozen TRAIN64 diagnostic tests its own mean-CE orientation prediction. A positive result there would not by itself prove held-out utility or calibrated confidence.

## Evidence and scope

`balanced-completion-check.py` evaluates the original ordered 8-by-8 circuit matrices at phi=pi/6,pi/3,2pi/5. It checks prepared-density equality, the five matched positive scores, coherent scores, the shared linear mapper, every common branch, actual EPS-normalized two-token logits and both binary/forty-class cross-entropies. Its [output](balanced-completion-check.json) has maximum circuit-equation error 4.45e-16 and loss-identity error 4.45e-16. This is a dense mathematical sanity check, not execution of native Torch training, natural-text evidence or another lifecycle canary.

One bounded A6API gpt-6.2-astra/high [scientific consultation](balanced-completion-consultation.json) suggested the positive-fidelity collision and binary conditional argument. The primary derivation independently reconstructed the circuit, supplied the shared-mapper realization, extended the loss identity to the exact forty-class readout and performed the dense check. The consultation is not release approval or independent empirical validation.

The process-fidelity identity is established prior art: [Nielsen, quant-ph/0205035v2](https://arxiv.org/abs/quant-ph/0205035v2), read in all three parsed text pages; see the [compact reading receipt](process-kernel-reading-nielsen.json). The broader distinction between state-map rank and readout sensitivity is also prior art: [Park et al., 2610.29186v1](https://arxiv.org/abs/2610.29186v1), introduction, section2.1 and section3 theorem1/corollaries2–4 scope-read; see the [receipt](readout-identifiability-reading.json). Neither reading establishes priority for this particular attention construction. No exhaustive nearest-method review or novel generic kernel theory is claimed.

The construction is separable at the relative-unitary level and exactly classically computable. It does not show entanglement necessity, hardware acceleration, impossibility for alternative process observables, generic state-map regularity or natural-language information separation. Its substantive candidate contribution is the explicit fixed-encoding collision at positive process fidelity and the precise route from return-path interference to normalized value contrast, together with the distinction between orientation benefit and predictive loss.

The running frozen study still decides utility and matched-control gates. Only their conjunction permits the unchanged, already frozen final-checkpoint TRAIN64 branch diagnostic. Failure closes this exact trained recipe without rescue. Even success plus positive TRAIN64 Delta would provide exposed-cohort single-seed discovery and within-checkpoint TRAIN mechanism evidence; appropriate independent evidence and substantiated nearest-method distinction remain necessary before L5 acceptance.
