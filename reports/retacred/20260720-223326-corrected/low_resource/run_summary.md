# Relation Run Summary

Run directory: `/root/projects/Q-Attention/runs/retacred_low_resource_gpu/20260720-222730`

Reported split: `test`

| variant | macro_f1 | accuracy | macro_precision | macro_recall | loss | delta_macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.121923 | 0.148234 | 0.123908 | 0.210806 | 4.748732 |  | held-out test split |
| classical_steering | 0.118807 | 0.146296 | 0.121077 | 0.205510 | 4.757472 | -0.003115 |  |
| quantum_steering | 0.119198 | 0.146892 | 0.121367 | 0.206831 | 4.753557 | -0.002724 |  |
| supervised_quantum_steering | 0.121923 | 0.148234 | 0.123908 | 0.210806 | 4.748732 | 0.000000 | standalone label-aligned quantum projector, layer-specific, coordinate gains selected on validation, active_layers=0, validation CI rejected proposed steering |
| spectral_sweep_best | 0.119198 | 0.146892 | 0.121367 | 0.206831 | 4.753557 | -0.002724 | quantum, hard_topk, rank=16, threshold=0.5, sharpness=8.0, gain=0.25, selected on validation |
| adaptive_routing | 0.120334 | 0.148085 | 0.122198 | 0.208787 | 4.752743 | -0.001589 | mean_entropy=0.530686 |
