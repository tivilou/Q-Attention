# Relation Run Summary

Run directory: `runs/retacred_low_resource_gpu`

| variant | macro_f1 | accuracy | macro_precision | macro_recall | loss | delta_macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.186459 | 0.249023 | 0.218157 | 0.216478 | 3.683047 |  | best validation checkpoint |
| classical_steering | 0.189302 | 0.250000 | 0.220974 | 0.219195 | 3.689244 | 0.002843 |  |
| quantum_steering | 0.187363 | 0.249634 | 0.218916 | 0.216962 | 3.687143 | 0.000904 |  |
| spectral_sweep_best | 0.189653 | 0.249512 | 0.222003 | 0.219516 | 3.702156 | 0.003194 | classical, hard_topk, rank=8, threshold=0.5, sharpness=8.0, gain=0.5 |
| adaptive_routing | 0.187172 | 0.249268 | 0.218758 | 0.217176 | 3.686388 | 0.000713 | mean_entropy=0.576293 |
