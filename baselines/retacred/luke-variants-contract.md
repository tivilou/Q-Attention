# LUKE variants: verified sources and missing Re-TACRED ports

**Not_verifiable / no training adapter.** Sources are not interchangeable:

- [CoRE paper](https://aclanthology.org/2022.naacl-main.224.pdf) reports LUKE
  variants on Re-TACRED. [Released CoRE source](https://github.com/vanoracai/CoRE)
  is pinned at c98d69472110d2514458e3b58a6a82f9793974aa.
- [Original CFIE paper](https://arxiv.org/pdf/2109.05213v1), section1, explicitly
  links [HeyyyyyyG/CFIE](https://github.com/HeyyyyyyG/CFIE). Its default branch
  is **NER**. The **RE** branch is pinned at17583143a4d70eb4c12e07461620b24dfe376198.
  Twelve selected public source files are hash/Git-blob locked. Root license is
  Apache-2.0; third-party notices remain unreviewed, and no upstream file is
  vendored or executed as a full module.
- [Official LUKE training directory](https://github.com/studio-ousia/luke/tree/251654548a15b9a8ddc708a53b715bdf3a112102/examples/relation_classification)
  was discovered and pinned separately. It supplies a TACRED training route,
  not evidence that the CoRE authors' Re-TACRED/CFIE port has been recovered.

## Bind each reported score to the correct table

CoRE paper Table3 reports **micro-F1**; Table2 reports **macro-F1**. Section4.1
reports the **median of five training runs**, not one seed, a mean or mean±SD.
All numbers below are paper reports, not project measurements.

| Re-TACRED row | Micro-F1 (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| Unmodified LUKE | 90.3 | 80.2 |
| LUKE + Resample | 90.5 | 80.5 |
| LUKE + Focal | 90.4 | 80.3 |
| LUKE + CFIE | 90.5 | 80.4 |
| LUKE + CoRE | 90.9 | 81.6 |

Exact seed IDs, checkpoint/split hashes and evaluator binding are not recovered.
Preserve historical selection rows, but do not rank a newly executed challenge
macro score against these full-test micro reports.

## Released CoRE is a challenge analysis, not full-test reproduction

The inspected `core.py`:

1. Loads `preprocess.data` via joblib, with gold labels, type constraints and
   cached label-only outputs whose production code is not released here.
2. Loads `studio-ousia/luke-large-finetuned-tacred`, not a Re-TACRED-trained
   forty-class checkpoint, and runs repository `test.json` (TACRED).
3. Computes full-context and entity-only predictions, then **selects examples
   whose entity-only argmax differs from the gold label**. Every later score
   uses that challenge subset, not the full test denominator.
4. Uses `getF1`: macro averaging over a numeric gold min..max label range,
   dropping the first range entry. Absent labels initialize precision/recall
   to1. This is not our non-null benchmark micro-F1 or fixed40-label macro-F1.
5. Applies fixed coefficients and cached type constraints to probability
   outputs. The source uses addition coefficients(-1.6,+0.1), while the paper
   subtracts two λ terms. Do not silently equate signs, the released example's
   coefficients or its challenge tuning with the paper's dev-selected recipe.

Gold-dependent challenge selection is a legitimate *analysis definition*, not
proof that the paper's full-test result is invalid. It is nevertheless outside
our canonical full-test denominator. No model, pickle or raw test dataset is
loaded during this audit. Two small, hash-checked AST slices execute only on
synthetic vectors to demonstrate scorer/denominator differences.

CoRE is an inference-time correction to an already trained model. The paper
selects its two coefficients on validation data over[-2,2] in0.1 increments;
test must not select coefficients. Entity-type candidate restriction is part
of the paper's method, not a project repair to add arbitrarily.

## CFIE is not the same correction, or a mask-only callback

CFIE's paper uses a causal graph with sentence/relation and NER/POS features,
edge-specific classifier losses, dependency-based counterfactual generation
and a direct-effect score. Exact α/β and branch semantics must be resolved
from the complete configuration, not a universal subtraction default.

The pinned RE source defaults to **NYT24**, with GloVe/contextual GCN and a
BERT branch. Inspected code and the complete tree do not establish a LUKE
implementation or the Re-TACRED recipe behind CoRE's comparison row.

- GCN and BERT use different auxiliary-loss combinations when `cfgen=false`.
- GCN's inspected TDE inference branch actively adds an `amp_intervention_logits`
  term and optionally adds a separate moving-average intervention effect.
  BERT has a different subtraction branch. Neither branch is silently chosen
  as the original LUKE port just because a flag says TDE.
- Context/dependency/counterfactual-generation/multiple-intervention flags,
  vocabulary, parser, label mapping, initialization and dev selection matter.
- A class-aware sampler exists in the original repository, but that does not
  bind its distribution/budget to CoRE's LUKE+Resample experiment. The pinned
  loss-file fetch failed; no textbook focal gamma/alpha is supplied as original.

Two individual source requests(eval/loss) failed TLS handshakes and were not
retried. Twenty-four bounded public HTTP requests, twenty-two successful;
no browser, raw dataset, checkpoint or weight download. Source recovery is
partial, not an importability or numerical model-fidelity certificate.

## LUKE foundation discovered; admission remains separate

The pinned official relation README trains from `studio-ousia/luke-large` with
AllenNLP and entity-aware attention. Its base configuration shows five epochs,
AdamW1e-5/decay.01, bias/LayerNorm exclusions, warmup.06, physical batch4,
accumulation8, dev micro-F1 and seed0. This is a **discovered official TACRED
configuration**, not automatically the CoRE authors' complete Re-TACRED recipe.
Encoder weights, entity vocabulary/IDs, entity spans, classifiers and metrics
must be audited before a declared canonical40-class adaptation.

Required future order:

1. Recover reader/model/entity-aware-attention and checkpoint identity for the
   unmodified LUKE foundation; freeze canonical converter and label boundary.
2. Pass fixed-input forward/loss/gradient/update and train/test numeric trace
   gates, separately from full-weight CUDA validation and common training.
3. Only then implement paired CFIE/CoRE/resampling/focal mechanisms with their
   exact losses, inputs, tuning and declared assumptions. Missing ports remain
   `paper_interpretive` at most, outside strict comparison until evidence gates.

Case studies must capture actual word **and entity** embeddings/attention,
factual and every counterfactual input/output, masks/graphs, component scores,
auxiliary losses and gradients. Tuning reports must identify dev-only data and
coefficients; full-test and challenge analysis have separate denominators.

## Offline audit

```bash
python -m baselines.retacred.luke_audit
python -m baselines.retacred.luke_audit --verify-cache <public-evidence-cache>
```

Requires the existing pinned CoRE source cache; never clones/downloads. Exit2
is expected: not_verifiable, no model jobs. Wrong source or tampered cache exits1.
Explicit LUKE variants are blocked before worker launch; default five remain.
