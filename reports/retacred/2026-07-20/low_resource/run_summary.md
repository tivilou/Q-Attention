# Relation Run Summary

Run directory: `runs/retacred_low_resource_gpu`

Reported split: `test`

| variant | macro_f1 | accuracy | macro_precision | macro_recall | loss | delta_macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.113633 | 0.161574 | 0.111932 | 0.218673 | 4.846012 |  | held-out test split |
| classical_steering | 0.113349 | 0.160158 | 0.111306 | 0.218801 | 4.855062 | -0.000284 |  |
| quantum_steering | 0.115006 | 0.161276 | 0.113299 | 0.219407 | 4.854721 | 0.001373 |  |
| supervised_quantum_steering | 0.113633 | 0.161574 | 0.111932 | 0.218673 | 4.846012 | 0.000000 | standalone label-aligned quantum projector, layer-specific, coordinate gains selected on validation, active_layers=0, validation CI rejected proposed steering |
| spectral_sweep_best | 0.113010 | 0.158518 | 0.110672 | 0.218586 | 4.874581 | -0.000624 | classical, hard_topk, rank=8, threshold=0.5, sharpness=8.0, gain=0.5, selected on validation |
| adaptive_routing | 0.114107 | 0.160754 | 0.112394 | 0.218523 | 4.851155 | 0.000474 | mean_entropy=0.552276 |
