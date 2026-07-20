# Relation Run Summary

Run directory: `runs/retacred_full_gpu`

Reported split: `test`

| variant | macro_f1 | accuracy | macro_precision | macro_recall | loss | delta_macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.245662 | 0.668207 | 0.310514 | 0.230407 | 1.815330 |  | held-out test split |
| classical_steering | 0.242716 | 0.667983 | 0.308256 | 0.227185 | 1.822711 | -0.002946 |  |
| quantum_steering | 0.244105 | 0.668207 | 0.312138 | 0.227973 | 1.818925 | -0.001557 |  |
| supervised_quantum_steering | 0.245662 | 0.668207 | 0.310514 | 0.230407 | 1.815330 | 0.000000 | standalone label-aligned quantum projector, layer-specific, gain=0 selected on validation |
| spectral_sweep_best | 0.245542 | 0.668132 | 0.311081 | 0.230151 | 1.815869 | -0.000120 | classical, band_pass, threshold=0.5, sharpness=8.0, gain=0.1, selected on validation |
| adaptive_routing | 0.245131 | 0.668132 | 0.311297 | 0.228999 | 1.816612 | -0.000531 | mean_entropy=0.595529 |
