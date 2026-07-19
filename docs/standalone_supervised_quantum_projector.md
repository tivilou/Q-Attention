# Standalone Supervised Quantum Projector

## Positioning

The primary Q-LASS method builds a quantum projector independently. It does not require a classical projector in its forward or construction path.

The original steering mechanism remains unchanged:

```text
k' = k + g P_q k
```

The optional classical-plus-quantum residual projector is an ablation only.

## Relation Representation

For each steerable attention layer, subject and object key vectors are mean pooled and converted into a relation feature:

```text
z = [k_subject, k_object, k_subject - k_object, k_subject * k_object]
x = 0.5 * (k_subject + k_object)
```

`z` is encoded by the quantum circuit. `x` is the aligned key-space vector used to lift the learned quantum geometry back into the steering space.

## Quantum Circuit

The Torch statevector prototype uses:

```text
deterministic angle projection
data re-uploading RY/RZ layers
ring-CNOT entanglement
trainable rotation scales and biases
fidelity kernel K_q
```

Circuit parameters are optimized on a class-balanced training subset by centered kernel-target alignment:

```text
L_q = 1 - alignment(K_q, K_label)
```

The best alignment checkpoint is retained.

## Standalone Projector

After circuit fitting, a class-balanced set of relation samples produces:

```text
Omega_q = X^T H K_q H X
P_q = U_q f(Sigma_q) U_q^T
```

No classical covariance projector appears in this construction.

## Layer-Specific Projectors

The default formal configuration trains one standalone quantum projector per
steerable key layer. Each layer receives its own relation-key samples, quantum
feature-map parameters, kernel alignment diagnostic, and spectral projector:

```text
P_q^(l) = U_l f(Sigma_l) U_l^T
```

The adapter applies `P_q^(l)` only to the matching key module. A single shared
projector remains supported for backward-compatible ablations.

## Gain Selection

The formal steering stage never tunes on the test split. Candidate gains are
evaluated on `valid.jsonl`, then frozen before one evaluation on `test.jsonl`.
The formal `coordinate` strategy selects a gain for each layer while all other
layers are held at zero. The legacy `shared` strategy remains an ablation.
Gain `0.0` is included. The validation split is internally divided into a
selection subset and an acceptance subset; the formal configuration requires
the positive paired-bootstrap interval on the acceptance subset before
accepting any nonzero steering.

## Toy Command

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/relation_real_smoke.json \
  --train_path examples/relation_toy_train.jsonl \
  --valid_path examples/relation_toy_valid.jsonl \
  --test_path examples/relation_toy_test.jsonl \
  --output_dir runs/qlass_toy \
  --device cpu \
  --stages baseline,supervised_quantum_projector,supervised_quantum_gain_selection
```

## Re-TACRED Debug Command

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_debug_gpu.json \
  --output_dir runs/retacred_qlass_debug \
  --device cuda \
  --stages baseline,supervised_quantum_projector,supervised_quantum_gain_selection
```

## Required Ablations

```text
classical covariance projector
untrained quantum projector
standalone supervised quantum projector
RBF or random-feature kernel projector
classical-plus-quantum residual projector
```

The main method claim depends on the standalone supervised quantum projector, not the residual ablation.

Build the residual ablation only after both source projectors exist:

```bash
python experiments/build_quantum_residual_ablation.py \
  --classical_projector_path runs/example/baseline/relation_projector.pt \
  --quantum_projector_path runs/example/baseline/relation_supervised_quantum_projector.pt \
  --output_path runs/example/baseline/relation_quantum_residual_ablation.pt \
  --alpha 0.5
```
