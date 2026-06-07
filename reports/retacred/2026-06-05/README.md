# Re-TACRED GPU Result Snapshot (2026-06-05)

Source branch: `origin/1.0` (`43f0677`, submitted by dzy958).

This directory contains sanitized experiment summaries only. Raw data, checkpoints,
projector weights, predictions, and full run directories are intentionally excluded.

## Layout

- `low_resource/`: sampled low-resource Re-TACRED run summaries.
- `full/`: full Re-TACRED run summaries.
- `logs/`: tail-style JSON logs copied from the collaborator run.

## Main Metrics

| split | variant | macro_f1 | accuracy | delta_macro_f1 | note |
| --- | --- | ---: | ---: | ---: | --- |
| low_resource | baseline | 0.186459 | 0.249023 |  | best validation checkpoint |
| low_resource | quantum_steering | 0.187363 | 0.249634 | +0.000904 | direct quantum projector |
| low_resource | spectral_sweep_best | 0.189653 | 0.249512 | +0.003194 | classical, hard_topk, rank=8, threshold=0.5, sharpness=8.0, gain=0.5 |
| full | baseline | 0.293637 | 0.659161 |  | best validation checkpoint |
| full | quantum_steering | 0.292732 | 0.659058 | -0.000905 | direct quantum projector |
| full | spectral_sweep_best | 0.294548 | 0.659365 | +0.000912 | classical, band_pass, threshold=0.5, sharpness=8.0, gain=0.1 |

## Readout

- Low-resource runs show a small but consistent positive signal.
- Full-data direct steering is negative; the best full-data gain comes from spectral filtering.
- Adaptive routing is close to uniform routing, so the next design pass should make the router more discriminative.
