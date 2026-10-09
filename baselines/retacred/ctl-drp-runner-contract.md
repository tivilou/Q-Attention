# CTL-DRP bounded runner / evidence lifecycle

This is a callable candidate lifecycle, not formal admission or a published
benchmark reproduction. CLI execution requires explicit `--tiny`; production
adapters fail the existing cohort admission gate before starting a job.
No real dataset receipt is present on the development server. The production
checkpoint/transport path needs independent review and real-data preflight.

## Portable bounded entry

`scripts/check_retacred_ctl_drp_runner.sh` resolves its repository root and
`PYTHON_BIN`, then python/python3. `--describe` performs no training/network/Git
action. Explicit `--tiny --cache CACHE --paper-source NATIVE --run-dir PRIVATE`
runs only two random tiny-model updates, with device cuda:0 by default or cpu
for diagnostics. Source/cache/weight paths are never hardcoded into this entry.
Optional `--pause-after 1` pauses after the first durable update and exits75;
repeat the same arguments plus `--resume` without pause-after to continue.
`--local-export-dir PRIVATE_REPORT` adds automatic private diagnostic whitelist
export on completion. It does not upload, commit or push.

## Checkpoint transactions

- The immutable contract includes physical data/H, description and source hashes,
  model/dropout configuration, physical micro-batch, recipe, sampled IDs,
  precision, environment and actual diagnostic budget. Formal budget remains
  the frozen complete ten epochs; tiny budget does not become a full-run claim.
- Initial generation stores exact RNG and an initialization signature, allowing
  reconstruction of cursor0 from the verified initializer without a redundant
  full-model file. Positive generations use the existing CPU diagnostic or
  strict single-visible-GPU FP32 backend; GPU UUID/runtime must remain identical.
- A generation writes state.pt and its hashed receipt, fsyncs them, renames the
  directory, then atomically publishes latest.json. Failed unpublished writes
  do not advance the completed-logical-update cursor.
- Two rolling generations are retained, plus source generations before/after
  the selected train event for replay. Full pretrained checkpoints are several
  GiB: budget sufficient disk on the collaborator, including one staging file.
- An advisory run lock prevents two owners. Checksum/contract drift fails before
  checkpoint restoration; there is no automatic fallback to an older cursor.
- Event files written before an unpublished checkpoint are quarantined on
  resume, never mistaken for durable samples or silently overwritten. Final
  evaluation failure resumes finalization from the last update, not training.
- CPU is diagnostic-only. CUDA scope is unchanged single GPU, FP32 AdamW,
  no AMP/topology migration. OOM exits with the last durable update intact;
  no new batch/chunk search is claimed here.

## Frozen observability and actual values

Before updates, freeze the first sample ID in lexical order from train and test.
The training sample is captured from its first real forward/backward event;
the test sample is captured at final post-update evaluation. No second forward
manufactures a historical training trace. Each sample retains every candidate
row, original sentence/tokens, entities with roles/types/spans, descriptions,
BPE tokens/IDs/masks/positions, correct and predicted relation.

Required numeric inventory derives from the implemented layer count, independent
of whichever keys happen to be captured: token IDs/mask; word, position and type
embeddings and combined embedding output; every layer's input, raw Q/K/V,
attention scores/probabilities, recombined context, attention-block output,
feedforward activation and layer output; raw first token, scalar scores and
candidate probabilities. Train additionally requires actual selected loss,
pre-update head, aggregate weighted-logical-batch head gradient and post-update
head. Other actual leaf I/O and activation gradients remain in detached audit
evidence. Parameter gradients are not falsely attributed to one selected sample.

JSON conforms to `sample-trace.v2`, with human-purpose/help text and explicit
producer input/output references. Numeric values stay in hash-checked private
.pt, not JSON previews. `ValueReader` verifies the artifact and serves arbitrary
prefix-axis matrix/vector/scalar slices bounded to4096 elements. Readiness opens
all required states, checks complete element counts, compares first/last reader
coordinates and actual layer/CLS/probability lineage. This is a verified local
reader, not a deployed portal route. Authenticated delivery/portal binding are
separate gates and are not claimed from local fixtures.

## Export / publication boundary

Diagnostic success writes DIAGNOSTIC_COMPLETE, never COMPUTE_COMPLETE or
RUN_COMPLETE. Formal computation would write COMPUTE_COMPLETE only after all
updates and train/test value readiness. Required authenticated hash/size delivery
uses the existing sharding uploader; pending manifests survive a failure and
export retry does not retrain. Only successful required delivery and atomic
whitelist export may produce RUN_COMPLETE.

The whitelist contains metrics.json, run_summary.json, data_identity.json,
data.sha256, data_counts.txt, run_summary.data, plus their export_manifest.json.
No raw samples, predictions, numeric values, checkpoints, token, environment URL
or arbitrary upload payload enters it. Diagnostic export is explicit, local-only
and rejected under public reports/. Every report file is bounded to1MiB and
hash-checked; repeated export is idempotent only for the same run identity.
This module makes no implicit Git commit/push. Formal one-command publication,
independent review, actual exchange and qualification binding remain separate
release tasks. Default-ready cohort remains five.
