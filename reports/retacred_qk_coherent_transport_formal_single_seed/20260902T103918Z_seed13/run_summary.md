# QK Coherent-Transport Re-TACRED Formal Single Seed

This is one complete seed-13 run under the frozen natural-task contract.

- candidate: `qk_coherent_transport`
- matched control: `classical_coherent_transport`
- parallel mode: `selector_parallel`
- model-parallel physical GPUs: `[]`
- hardware profile: `adaptive` (chunk=all/divisor=1 | physical_batch=256 | accumulation=1 | activation_checkpointing=false)
- selected physical GPUs: `[0, 1, 2]`
- candidate minus disabled test macro-F1: `0.003833`
- candidate minus matched test macro-F1: `-0.000604`
- practical gain gate: `true`
- matched comparator gate: `false`

The test split is evaluated only after training and validation selection. This single seed does not authorize multi-seed replication.
