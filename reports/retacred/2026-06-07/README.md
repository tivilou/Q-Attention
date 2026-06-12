# Re-TACRED GPU Result Snapshot (2026-06-07)

Source branch: `origin/1.1` (`66c3bbf`, submitted by dzy958).

This directory contains sanitized experiment summaries only. Raw data, checkpoints,
projector weights, predictions, and full run directories are intentionally excluded.
The duplicate `configs/*.log` files from the submitted branch are omitted because
`logs/` already contains the same run tails.

## Layout

- `low_resource/`: sampled low-resource Re-TACRED run summaries.
- `full/`: full Re-TACRED run summaries.
- `logs/`: tail-style JSON logs copied from the collaborator run.

## Main Metrics

| split | variant | macro_f1 | accuracy | delta_macro_f1 | note |
| --- | --- | ---: | ---: | ---: | --- |
| low_resource | baseline | 0.186459 | 0.249023 |  | best validation checkpoint |
| low_resource | quantum_steering | 0.187363 | 0.249634 | +0.000904 | centered quantum projector |
| low_resource | spectral_sweep_best | 0.189653 | 0.249512 | +0.003194 | classical, hard_topk, rank=8, threshold=0.5, sharpness=8.0, gain=0.5 |
| low_resource | adaptive_routing | 0.187299 | 0.249268 | +0.000840 | mean_entropy=0.576293 |
| full | baseline | 0.293637 | 0.659161 |  | best validation checkpoint |
| full | quantum_steering | 0.292732 | 0.659058 | -0.000905 | centered quantum projector |
| full | spectral_sweep_best | 0.294548 | 0.659365 | +0.000912 | classical, band_pass, threshold=0.5, sharpness=8.0, gain=0.1 |
| full | adaptive_routing | 0.292818 | 0.659058 | -0.000819 | mean_entropy=0.597061 |

## Readout

- Centered quantum kernels worked numerically: run logs report quantum kernel means close to zero.
- Routing is no longer near-uniform: mean entropy dropped to about 0.58-0.60 from the previous near-maximum value.
- Routing accuracy did not improve with the sharper router, so the next design pass should decouple expert selection from expert steering strength.
