# Exact source-kernel correspondence and transfer limits

Source: arXiv:2609.06016v1, PDF p8 Eq5/6, visually checked. This is known elementary algebra, not a new theorem or a native task experiment.

Write a single encoded feature as q(c)=cos(pi*c/2)|0>+sin(pi*c/2)|1>. The real inner product is q(c)^T q(c')=cos(pi*(c-c')/2). Tensor products multiply inner products. Hence the source's ideal squared-fidelity kernel is exactly

`K(c,c') = product_l cos(pi*(c_l-c'_l)/2)^2`.

Eq5 has no inter-qubit gates or trainable circuit parameters. This does not prove that a nonlinear product kernel is useless. It identifies the exact matched analytic computation needed to attribute any additional effect to quantum evaluation.

Hold raw centres, normalisation, K values, neighbourhood convention, ties and any random state fixed. Then Eq7 cohesion, Eq8 sparse graph, symmetric adjacency, distances, average-linkage decisions and later noise reassignment receive identical inputs under this formula. The ideal pipeline therefore produces identical outputs. Actual author-code settings and finite-shot execution were not inspected. Equality of ideal K is not a reproduction of a noisy implementation or a hardware runtime claim.

The source first compresses *features*, then applies a nonlinear embedding. It is generally different from a density-matrix average. For one feature, two centres 0 and1 average to1/2. At query1/2, encoding the centre gives fidelity1; the mean of endpoint density matrices is I/2 and gives fidelity1/2. This exact analytic example establishes noncommutation only. It is not a task case, a native circuit-port execution, a performance gain or a new theory result.

Source fidelity and cohesion concern cluster geometry. Native Q-RMEA instead learns a 3-qubit/two-layer entangling map and40 binary effects, normalises token scores with EPS and sums signed values. Source n-by-n-to-m-by-m Gram savings and cluster bridge avoidance do not imply a benefit for that fixed40-by-token attention computation. Temporarily excluding low-cohesion granules followed by reassignment also differs from permanently dropping attention evidence. No such modified model was implemented here.
