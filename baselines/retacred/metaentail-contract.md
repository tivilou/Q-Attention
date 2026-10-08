# MetaEntail paper-derived primitives

## Status and admission

Provenance: paper, not imported author code. The official implementation was
unavailable in the preceding bounded audit. This preparation implements data
and decision primitives, not a neural model or published94.3 reproduction.
Fidelity remains not_verifiable; strict comparison and formal runner admission
remain closed. No default model, task, data split or score is substituted.

The official ACL PDF is hash-locked in metaentail_paper.lock.json. Table4 supplies
all40 ReTACRED hypothesis templates and Table13 supplies the40x40 NLI target
matrix. The fidelity command regenerates PDF text with pdftotext and compares
all templates and1600 cells. Raw paper bytes stay in the private source cache.
Canonical numeric relation IDs match the pinned Improved RETACREDProcessor.

## Implemented mechanism

- General-domain Section5.3 uses FacebookAI/roberta-large-mnli and keeps entity
  surface forms. record_input preserves original tokens and inclusive role spans;
  it never substitutes entity types in the premise or hypothesis.
- The feasible type-pair filter is fit on training examples only. Each relation
  stores ordered subject/object type pairs actually observed for that relation.
  No global hand-written type constraints or test-label-based rescue is added.
- NLI targets: the true hypothesis is entailment; no_relation contradicts every
  positive relation and vice versa. All six family/identity classes mutually
  contradict; org:member_of and org:members contradict. Other off-diagonal
  combinations are neutral, not automatically contradiction.
- Inference reads only sample ID, tokens, entity spans and types. It does not
  read, copy, serialize or branch on the gold relation. Training target creation
  is a separate function and verifies the filter includes training gold.
- Each feasible hypothesis receives one three-way NLI probability vector.
  Only rows whose NLI argmax is entailment are eligible; among them, select the
  highest entailment probability. This is not argmax entailment across all rows.
- Empty feasible groups or no entail predictions remain explicit abstentions,
  with their reasons and complete candidate decisions retained. No synthetic
  no_relation prediction or gold-dependent post-processing is introduced.
- Filter serialization has a deterministic integrity checksum, immutable
  type-pair sets and a hash of the training id/type/relation projection. This
  projection is not a full dataset hash and not an authentication signature.

## Declared interface choices

- Premises and multi-token entity surfaces use literal space-joined raw tokens.
  No undocumented PTB conversion, marker insertion or truncation occurs here.
- Curly apostrophes become ASCII; subj/obj template fields become named format
  fields without changing words. org:dissolved intentionally omits the object,
  as printed in Table4; do not silently repair this template.
- The checkpoint's actual three NLI labels/order must be passed explicitly.
  We do not assume they match relation IDs or hardcode an entailment column.
- Exact probability ties use the first checkpoint NLI index, then the first
  canonical relation ID. Invalid, nonfinite, unnormalized or misaligned vectors
  are rejected, not normalized or repaired.
- Unknown type pairs remain empty groups; inputs with invalid/overlapping spans
  and duplicate training IDs stop with a diagnostic rather than being dropped.
- audit_decisions is an all-example diagnostic: every original example counts
  toward accuracy; a positive gold abstention contributes a false negative.
  Coverage/abstentions are explicit. It is not the recovered paper evaluator,
  not the final full-metric report implementation, and fabricates no loss.

## Unresolved before a neural runner

Table6 is captioned BioLinkBERT, not explicitly the general-domain RoBERTa
recipe. Its1024 limit must not be sent silently to a standard RoBERTa checkpoint.
Official checkpoint metadata is now pinned in metaentail_checkpoint.lock.json.
Six actual small files match provider Git blobs and SHA256; the config binds
three-way NLI order and512-token pair capacity. Real fast-BPE sentence-pair
boundary checks and explicit expanded-pair budget arithmetic are prepared in
metaentail_checkpoint.py. Weight hashes are provider metadata only; no neural
loading/fidelity claim follows. The later metaentail_assets.lock.json now binds
full SHA-verified safetensors and private nine-shard delivery; mnli_assets.py
provides atomic retrieval and metaentail_neural.py prepares a dedicated pretrained
NLI foundation. Actual neural runtime remains memory-gated and formal suite
admission stays closed. See metaentail-neural-contract.md for the
distinct paper facts, proposed interface choices and remaining runtime gates.
Actual NLI-head loading, source optimizer/schedule/model selection,
and the general-domain expanded-NLI training recipe still need binding.
Do not replace the pretrained MNLI head without a declared initialization law.

The paper says only gold/predicted entail instances map back for evaluation;
its abstention denominator still requires reconciliation with the all-example
non-null benchmark evaluator. Do not present a filtered positive-only score as
canonical F1. Independent mechanism review, neural forward/loss/gradient/update,
exact batch resume, actual numeric train/test case traces, full-weight CUDA
and published-score/multi-seed evidence remain pending.

## Bounded commands

`python -m baselines.retacred.metaentail --toy` executes synthetic primitive
fixtures only. Optional --output writes a new evidence file; existing paths are
not overwritten. No Torch, weight download, task dataset or network is used.

`python -m baselines.retacred.metaentail_fidelity --paper-cache PATH` verifies
cached official PDF bytes and actual Table4/Table13 values offline. Neither
command starts a neural worker, publishes reports or pushes Git.
