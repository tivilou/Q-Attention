# Q-CEASC Re-TACRED Formal Single Seed

This is one complete seed-13 run under the frozen natural-task contract.

- candidate: `q_ceasc_grouped_counterfactual`
- matched control: `classical_grouped_counterfactual`
- structural control: `random_grouped_counterfactual`
- parallel mode: `selector_parallel`
- model-parallel physical GPUs: `[]`
- hardware profile: `adaptive` (chunk=all/divisor=1 | physical_batch=128 | accumulation=2 | activation_checkpointing=true)
- selected physical GPUs: `[0, 1, 2]`
- candidate minus disabled test macro-F1: `0.000000`
- candidate minus matched test macro-F1: `-0.000113`
- candidate minus random-group test macro-F1: `0.000000`
- candidate relative gain vs disabled: `0.000000%`
- classical relative gain vs disabled: `0.063571%`
- L1 utility gate (strictly positive): `false`
- quantum-inspired relative-gain gate (>1%): `false`
- practical gain gate: `true`
- matched comparator gate: `true`
- random-group control completed: `true`

The test split is evaluated only after training and validation selection. This single seed does not authorize multi-seed replication.
