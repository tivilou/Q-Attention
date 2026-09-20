# Q-CEASC Re-TACRED Formal Single Seed

This is one complete seed-13 run under the frozen natural-task contract.

- candidate: `q_ceasc_counterfactual`
- matched control: `classical_counterfactual`
- parallel mode: `selector_parallel`
- model-parallel physical GPUs: `[]`
- hardware profile: `adaptive` (chunk=all/divisor=64 | physical_batch=32 | accumulation=8 | activation_checkpointing=true)
- selected physical GPUs: `[0, 1, 2]`
- candidate minus disabled test macro-F1: `0.000067`
- candidate minus matched test macro-F1: `0.000120`
- candidate relative gain vs disabled: `0.037374%`
- classical relative gain vs disabled: `-0.029677%`
- L1 utility gate (strictly positive): `true`
- quantum-inspired relative-gain gate (>1%): `false`
- practical gain gate: `true`
- matched comparator gate: `true`

The test split is evaluated only after training and validation selection. This single seed does not authorize multi-seed replication.
