# Directional completion access in the MQT attention component

Status: scoped source comparison and primary algebra, not novelty clearance, a task experiment or L5 acceptance. The running full recipe and conditional TRAIN64 diagnostic remain unchanged.

## What was read

[Molecular Quantum Transformer, arXiv:2503.21686v2](https://arxiv.org/html/2503.21686v2), Yuichi Kamata, Quoc Hoan Tran, Yasuhiro Endo and Hirotaka Oshima, describes its detailed attention update in Methods, “Molecular Quantum Transformer / Quantum Transformer,” in the paragraphs referring to Fig.7(b). The official abstract metadata and original HTML title, authors and v2 watermark were checked. The comparison below uses that text, not a visual inspection of the figure or an audit of source code, resource claims or experiments.

The described component prepares token j on an auxiliary register, applies a Query transformation followed by the adjoint Key transformation, then the adjoint embedding for token k. A trainable MPS ansatz maps the resulting register into an attention ancilla. A separately prepared value register and controlled SWAP implement the feature update. This retains a coherent intermediate register; it should not be described as an ordinary scalar prepared-state fidelity test.

## A fixed-pair access statement

Write U=Embed(x_j), V=Embed(x_k), S=Key†Query. Before the MPS, the auxiliary state in the described pair component is

\[
|r\rangle=V^\dagger S U|0\rangle.
\]

Fix the other input, Query/Key/MPS parameters, incoming feature register and value register. The entire remaining component is then a quantum channel on |r><r|. In particular, an attention-ancilla observable can be represented by an effect M on this register, giving

\[
p_M=\langle0|U^\dagger S^\dagger V M V^\dagger S U|0\rangle.
\]

This notation analyzes an observable; it does not assert that MQT measures and classically normalizes a scalar attention score at that point.

If a hypothetical right-completion change U→UR satisfies R|0>=e^(i gamma)|0>, the query-preparation density is unchanged. With the fixed-pair assumptions above, the whole downstream component is unchanged, for any M. In contrast, a key-side change V→VR changes the intermediate state to R†|r>, so an observable becomes <r|R M R†|r>. A general downstream MPS effect need not commute with R. Thus MQT cannot be labeled completion-blind on both sides merely because its Query and Key operations involve a quantum overlap circuit.

The current relative-return score p_C=|<0|(V†U)²|0>|² uses U again after the first relative operation. The [positive-fidelity construction](balanced-completion-theorem.md), with S=I, changes this score through a query-side right-Z completion even though U|0><0|U† and the stated positive process comparator are unchanged. This distinguishes the query-side input-use contract of the fixed pair just described. It does not establish that the entire MQT function class lacks such information.

## Limits that matter for the L5 claim

- The original MQT angle embedding uses RY gates. The hypothetical right-completion transformation is an input-port analysis; this reading does not prove it is realizable by a second choice of its original restricted input coordinates.
- In complete self-attention, a token also occurs in key, value, self-pair and later-layer roles. Holding those roles and the incoming register fixed is essential to the pair statement. No whole-model invariance follows from the query-only calculation.
- Independent training can change the encoders, downstream MPS and classical residual path. No trained-class separation, parameter-efficiency comparison or task advantage over MQT follows.
- The described MQT is a concrete prior for process-sensitive quantum attention. “First quantum attention to use a full encoding process,” “all prior quantum attention is state-only,” and a universal exclusion of process observables are unsupported.
- Repeated input-unitary use/data reuploading is a known general resource. The contribution being tested is the particular relative double-return interference mechanism, the positive-fidelity collision and its normalized value/loss connection. Priority against the closest repeated-query attention designs remains unresolved.

The practical next action remains the original full-study comparison. A failed utility/control conjunction closes this exact trained recipe. A passing conjunction permits the already frozen TRAIN64 orientation diagnostic, not a claim of broad superiority or L5 acceptance.
