# Sequence Routing H3/Mix — recovery and admission audit

This is a **source audit, not an implemented/admitted model**. The cohort keeps
the historical row, but its reported 92.2 must not rank as same-metric evidence.
No new result was measured and no formal training ran.

## Locked provenance

- Active source: [author repository](https://github.com/bahramiramazan/re-representation/tree/cc95c4ad20b9ed46b978ffa4e88e31ce28c7ed20).
- Historical `Abstraction.py`: commit `6065881f5e55212d1ebddd3813218cafc7b1718c`,
  blob `59c8458bf6467ada925c1f9f6d15d4dbcdd1007c`, SHA-256
  `df4c064ee65ba840e5edbb91df89a4dc0672a990e3eaf8cd8d1e0869767d82dc`.
  Recovery is separate from the active checkout; do not transplant or execute it.
- [Publisher paper](https://aclanthology.org/2025.icnlsp-1.31/), Table5 and AppendixA.
- Public recovery inspected 16 of29 default-branch historical trees, one branch,
  zero tags and zero public forks. Path histories for the two Re-TACRED metadata
  files and root `LICENSE` returned no commits. That is bounded negative
  evidence, not proof that the files never existed in private/uninspected refs.
- Main repository license remains unresolved. The external Heinsen repository
  declares MIT, but upstream does not pin the dependency version; this does not
  supply a license or experiment version for Re-Representation itself.

## Confirmed source path and corrected findings

- `Experiments.experiment_run(..., 'four')` selects only `head_3`, uses Mix,
  sets H3 output512, batch24, and5epochs. `get_settings` initially sets H3 size41,
  **but the active experiment overwrites it to512**. Classifier output remains41;
  canonical Re-TACRED and the paper's dataset table have40 classes.
- `rc_data.prepare_data` constructs Mix using both entity surface forms and
  annotated types inside entity markers. Their order/punctuation is not
  identical to the paper's illustrative string; retain the released source,
  not a silently rewritten textual illustration.
- `RC_Model.Relation_Classifier_Model.body` obtains all25 RoBERTa hidden states,
  stacks them across depth; H3 normalizes/transforms/flattens position+depth,
  then performs three EfficientVectorRouting stages, followed by the source MLP.
- The H3 path uses relation cross-entropy, Adam LR1e-5, gradient clip.5 and a
  LinearLR schedule. Exact paper replay is not established: AppendixA describes
  H3 batch64/6epochs, whereas the released experiment uses24/5. Physical
  micro-batching is not permission to conceal this logical-budget difference.
- The two missing relation JSON files are **outputs of preprocessing**, not
  necessarily permanently unavailable artifacts. However, the clean-checkout
  factory calls `set_tokenizer` and reads the counter before preprocessing
  generates it. Exact bootstrap policy and first-occurrence relation/token order
  have not been certified. No arbitrary generated mapping is accepted as the
  original experiment mapping.
- `Abstraction.py` was recovered from official history; the current missing
  wildcard import remains unmodified. Its historical presence alone does not
  establish compatibility or fix the rest of the recipe.

## Metric boundary

The released `Train_Eval.rc_eval` accumulates all examples (including null
relations), calls sklearn micro-F1 without a labels filter, and returns that
value. For single-label classification this equals accuracy. The project's
benchmark micro P/R/F1 excludes `no_relation`.

An independent fixture with9 correct null predictions and1 missed positive
gives source all-label micro-F1/accuracy .9, but benchmark non-null micro-F1 0.
The two scores cannot be converted without class-level predictions/confusion
counts. The paper calls92.2 micro-F1 and80.1 macro-F1; **the exact binding of
that published score to the later released evaluator is not certified**.
Record the mismatch and retain the historical score only as reported evidence,
not as a verified non-null benchmark ranking or a project result.

## Runnable bounded commands

```bash
python -m baselines.retacred.sequence_resources
python -m baselines.retacred.sequence_resources --verify-only
python -m baselines.retacred.sequence_audit
```

Resource recovery makes one public HTTPS request, checks size/Gitblob/SHA256,
parses but never executes the file, and keeps it under private `.external/`.
`--verify-only` is offline/read-only. The audit is offline, rejects source/hash
drift, prints JSON and **exits2 for not_verifiable**; that expected code is not
a training crash. `--models sequence_routing_h3` still fails before job launch.
Neither command installs packages, allocates a model, writes data, changes the
active source or promotes the candidate.

## Remaining admission gates

Declare and resolve the40/41 output boundary, paper/source budget discrepancy,
metadata bootstrap/token order, dependency revision and main-source license.
Then implement a thin wrapper and validate active forward/loss/gradients/update,
scheduler, numeric train/test traces and exact checkpoint recovery. Full-weight
CUDA/common-protocol evidence stays separate from all static/CPU fixture checks.
