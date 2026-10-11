# CTL-DRP task adapter — paper-interpretive, not formal admission

`ctl_drp_task.py` binds physical train/dev/test files before fitting H, keeps
candidate-specific joint encoding and the scalar Eq2 head, and calls the shared
40-class classification scorer. The normal reference trainer/queue still does
not dispatch CTL-DRP. Default-ready remains five models. No formal run is started
by the adapter's descriptor or its bounded tests.

## Physical qualification boundary

The adapter consumes an **already qualified** external receipt; it does not
qualify a benchmark by assigning the word qualified to arbitrary files. The
owning qualification process must issue the following task-specific projection:

```json
{
  "schema": "ctl-drp-physical-splits.v1",
  "status": "qualified",
  "task": "retacred_sentence_relation_40",
  "canonical_labels": ["all 40 labels in canonical ID order"],
  "synthetic": false,
  "splits": {
    "train": {"file": "train.json", "format": "json_array", "sha256": "SHA256", "count": 58465},
    "dev": {"file": "dev.json", "format": "json_array", "sha256": "SHA256", "count": 19584},
    "test": {"file": "test.json", "format": "json_array", "sha256": "SHA256", "count": 13418}
  }
}
```

The example's labels/hash slots are explanatory, not a runnable qualification.
JSONL is accepted only with explicit `format: jsonl`. Filenames are relative to
the receipt directory. Symlinks, traversal, repeated physical files, changed
hash/count, duplicate sample IDs, invalid spans/types/labels, and incomplete real
splits fail before model loading. `synthetic: true` is permitted for small fixture
tests only and is rejected by `from_assets`. No real receipt/data is available
on this development server yet; the actual qualified full-data gate is pending.

H is derived only from the physically bound **entire train split**. Its separate
binding has physical identity true; the historical primitive flags stay unchanged.
Unknown heldout entity-type pairs fail preflight, rather than silently inserting
gold or predicting a forced relation. Known pairs with an excluded heldout gold
remain in all classification metrics. Their gold probability is zero, so true
mean NLL is infinite: JSON loss is null with explicit loss-status and excluded
indices/count. No epsilon repair or covered-subset loss is presented as full loss.
The common scorer now accepts explicit null loss, preserving finite-loss callers.

## Frozen recipe and objective

- Effective batch **512 sentences**, LR1e-5, ten epochs; these paper numbers are
  interpreted as sentences, not expanded candidate pairs.
- AdamW (.9,.999), eps1e-8, decay0, constant LR/warmup0, no clip, FP32,
  non-foreach/non-fused; final-epoch selection, seeds13/29/53. These missing-paper
  details are our explicit frozen choices, not recovered author settings.
- Physical micro-batches are fixed sentence partitions. Each loss is weighted
  by physical-sentence count / actual logical-batch count, including the short
  final update. Candidate count never weights a sentence's contribution.
- `UpdatePlan` regenerates each epoch's shuffled sample indices from local
  Python Random(seed+epoch), with a completed-logical-update cursor. It consumes
  no global dropout RNG and neither drops nor repeats a tail sample.
- Real factory checks all encoded candidate lengths/markers before loading the
  verified pinned pretrained backbone. Expanded features are lazy, not permanently
  cached for every sentence and relation.

Dropout means changing physical partition is **not** bit-exact; micro-batch is
bound in the contract. Only existing diagnostic checkpoint backends and the new
CPU next-plan replay are tested here. Do not claim a formal durable pause/resume
lifecycle or GPU topology/AMP migration from this adapter alone.

## Evidence boundary / remaining work

Tests use synthetic physical files and actual random tiny RoBERTa on CPU/CUDA,
not complete task training. Numerical train/test values are detached from real
fixture events and safely roundtripped; this is not a full formal Case Study
contract or delivery record. Earlier full24-layer supervised GPU evidence is
reused, not rerun or substituted with this tiny fixture.

Before collaborator release: bind the real qualification projection; implement
the formal runner's immutable cursor/contract checkpoint, predeclared selected
sample full-value traces, private tensor delivery/export and report lifecycle;
obtain independent mechanism review. Unsupervised asset/runtime remains pending.
Do not change cohort admission merely because this descriptor exits successfully.
