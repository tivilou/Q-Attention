# Adaptation boundary

## Allowed and declared

- Canonical JSONL tokens/entity spans/types -> upstream record fields;
  end-exclusive spans become inclusive only at that format boundary.
- Map output metrics to canonical names without changing predictions.
- Resolve data, Python, source and checkpoint paths relative to the checkout.
- Device placement and dynamic independent task assignment; one heavy worker
  per GPU. No distributed gradient averaging between separate seeds/models.
- Finalize dynamically added typed markers before optimizer construction.
- CPU-detached observers of actual train/test sample tensors; no replay is
  presented as an original training event. Runtime trace transparency is tested.
- Separately declared common optimizer/update/selection protocol, atomic
  checkpoints, complete metric accounting and private artifact transport.
- Explicit optional activation checkpointing, fixed for a run and its resume.

## Forbidden

- Replace subject/object marker pooling with CLS/mean pooling, replace the
  classifier/loss, change entity markers or silently truncate an entity away.
- Choose a checkpoint, tune a threshold, constrain a label by entity type,
  repair outputs, filter failures or select best examples using test targets.
- Fill unavailable PTR/Sequence Routing/CTL/LUKE paths with an ordinary encoder
  while retaining their names or reported scores.
- Change physical partition, data/source/weights or precision across a strict
  resume; overwrite another run, include private samples/weights in Git,
  or terminate unrelated GPU processes.

`postprocessing_rules` is empty. Argmax is the source classifier readout, not
an evaluation-time rule added by the adapter.
