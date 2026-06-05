# Relation Run Summary

Run directory: `runs/retacred_full_gpu`

| variant | macro_f1 | accuracy | macro_precision | macro_recall | loss | delta_macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.293637 | 0.659161 | 0.372802 | 0.269468 | 1.819823 |  | best validation checkpoint |
| classical_steering | 0.291995 | 0.659314 | 0.370312 | 0.267762 | 1.825692 | -0.001641 |  |
| quantum_steering | 0.292732 | 0.659058 | 0.372296 | 0.267948 | 1.823068 | -0.000905 |  |
| spectral_sweep_best | 0.294548 | 0.659365 | 0.373874 | 0.270224 | 1.820136 | 0.000912 | classical, band_pass, threshold=0.5, sharpness=8.0, gain=0.1 |
| adaptive_routing | 0.293268 | 0.659875 | 0.371863 | 0.268851 | 1.822273 | -0.000369 | mean_entropy=1.382794 |
