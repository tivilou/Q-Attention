# CTL-DRP: recovered paper mechanism and explicit implementation gaps

## Paper identity and evidence grade

[Sentence-Level Relation Extraction via Contrastive Learning with Descriptive
Relation Prompts](https://arxiv.org/abs/2304.04935v1), Jiewen Zheng and Ze Chen.
Paper provenance, not source-faithful reconstruction. TableVI reports
Re-TACRED P/R/F1 percentages91.3/91.9/91.6 for sup and91.2/91.2/91.2 for unsup.
These are averages over several unspecified random seeds, not one seed,
mean-plus-standard-deviation data or project measurements. Null-class scorer
denominator and original evaluator are not recovered from the micro-F1 label.

## Active mechanism, sectionsII-B throughII-E and equation2

1. Preserve the sentence and entity roles; surround subject/object spans with
   four newly registered tokens u1/u2 and u3/u4, with role/type annotations
   after each entity inside parentheses. This differs from punctuation markers.
2. Obtain a candidate set from the paper's H(relation, subject_type, object_type).
   Include NA unconditionally. The full H table/producer is unspecified; the
   appendix descriptions are not a recovered finite NER-type restriction table.
3. For every permitted candidate, combine marked sentence with its narrative
   relation prompt and description. NA uses its negative narrative alone.
   Both sentence and relation enter the encoder together; cache-once sentence
   embeddings followed by a separate relation dot product are not equivalent.
4. Read the first subtoken of the final Transformer layer. A single shared
   affine scalar head scores each candidate-specific representation.
5. Equation2 normalizes exp(score) over this sentence's permitted candidates
   plus NA. Maximize the gold probability; inference takes the largest score.
   This is not an independent sigmoid/BCE, cosine/temperature SimCSE objective,
   global in-batch negative denominator or a40-output classifier.

For candidate i, z_i = W dot h_i + b, p_i = exp(z_i)/sum_j exp(z_j),
and the translated per-example negative-log-likelihood is -log(p_gold).
The shared scalar bias cancels exactly from probabilities; its gradient is0.
Head gradients are sum_i(p_i - indicator_i_gold) h_i; representation gradients
are (p_i - indicator_i_gold) W. A candidate set containing only NA has probability1
and loss/gradient0 for NA. A missing gold candidate is a provenance problem to
report; do not reinsert it using the sample label or silently skip the sample.

`candidate_reference` implements only this fully specified scalar equation on
supplied finite embeddings and an explicit candidate set. It does not compute
the encoder, infer H, choose descriptions/checkpoints, train or admit the model.
All tied maxima are returned as a set; the paper's tie-breaking law is unbound.

## Sup versus unsup and recipe

SectionIII-B names sup-smi-roberta-large and unsup-smi-roberta-large and cites
SimCSE. Preserve those literal strings. The intended checkpoint mapping to
particular public SimCSE IDs/revisions must be resolved separately, not silently
replaced by generic RoBERTa-large or MNLI. The suffix distinguishes initialization;
both CTL-DRP variants use the same supervised gold-relation objective downstream.

Bound settings: batch512, learning rate1e-5, epochs10, several-seed mean. The unit
of batch512 (sentences versus expanded pairs), accumulation, optimizer details,
scheduler/warmup, sequence length/truncation, dropout, special-token initialization,
pooler/projection use, dev checkpoint selection and actual seed list are unbound.
Do not equate every setting to upstream SimCSE merely because the paper describes
its batch and LR as shared with pretraining.

## Relation-description and data boundary

Appendix TableIX has46 descriptions spanning TACRED/Re-TACRED, not40 canonical
classes. Offline native-TeX parsing plus declared aliases covers all40 canonical
labels in pinned Improved prepro.py without executing it. Hyphens are mapped to
underscores; org:top-members explicitly maps to org:top_members/employees and
no-relation to no_relation. Six noncanonical entries remain visible as extras;
do not add them as output classes. Description text stays in the private original
paper/cache and is not a hand-rewritten release asset.

The alias boundary validates identity/coverage only, not type restrictions, exact
tokenizer bytes or checkpoint fidelity. Published58465/19584/13418 counts and40
classes are paper facts, not our data hash certificate. No raw dataset was fetched.

## Source recovery and next gate

Selected repository head remains README-only. Fresh public metadata yielded one
branch, no tags/forks, five commits; their five nontruncated trees contain only
README and a historical1-byte dataset placeholder. No placeholder bytes fetched.
This is bounded public evidence, not proof about private repositories or author
identity. Nothing downloaded is imported as model source.

Portable check script verifies12 originals/native TeX/cohort/source-label hash
offline and exits2/recipe_neural_unresolved. Invalid evidence exits1; descriptor
exits0. No network/weights/Git/train action. All fifteen rows/ready-five retained,
both CTL-DRP rows remain blocked and rank-ineligible. Independent review and
assumption sensitivities are required before any materially underspecified neural
paper reconstruction is certified. Next: resolve explicit H producer/provenance and remaining paper recipe; exact
checkpoint mapping still needs author-level evidence before neural admission.


## Official SimCSE candidates: checkpoint/token/readout boundary

The pinned [official SimCSE repository](https://github.com/princeton-nlp/SimCSE/tree/49fa580a853752ede55b8c76d9debf748e214d3f)
links both public RoBERTa-large candidates. `ctl_drp_simcse.lock.json` binds 24
public metadata/source/tokenizer originals; small-file local SHA256 and Git blob
hashes are verified. Nothing fetched is imported as model code.

| Variant | Official candidate | Observed immutable revision | Weight status |
|---|---|---|---|
| sup | princeton-nlp/sup-simcse-roberta-large | 96d164d9950b72f4ce179cb1eb3414de0910953f | provider LFS hash only |
| unsup | princeton-nlp/unsup-simcse-roberta-large | d3f863b476c59b0673264042f159cea15842e265 | provider LFS hash only |

These are **official candidates**, not author-confirmed CTL initialization
identities. The CTL paper's literal `smi` names remain unchanged; correcting them
to `simcse` and selecting an exact revision would be an explicit assumption.
No large checkpoint is downloaded, deserialized, resized or numerically loaded.
Both candidate `.bin` sizes are 1,421,571,527 bytes, with distinct provider hashes.

Both configs report RobertaModel, 24 layers, width1024, vocabulary50265,
positions514. Their vocab, merges, tokenizer_config and special-token-map bytes
are identical; token IDs are <s>0, <pad>1, </s>2, <unk>3, <mask>50264. Public
tokenizer maximum is512 and add_prefix_space isfalse. The real byte-BPE fixture
checks single/pair boundaries, leading space, Unicode and lengths512/513 without
choosing truncation. It is not a recovered CTL prompt-joining convention, marker
registration/initialization, or original Transformers-version equivalence.

Official sentence-embedding evaluation recommends pooled linear+tanh `cls` for
sup and raw `cls_before_pooler` for unsup. The convenience tool also defaults to
L2 normalization. **Those are not the CTL downstream readout.** CTL sectionII-B
specifies the first subtoken of the final Transformer layer for both variants:
`last_hidden_state[:,0,:]`, then its own scalar head. Importing SimCSE tool output,
`pooler_output`, mean pooling or normalization silently changes this mechanism.
The source converter renames `mlp` keys to `pooler` and removes the backbone
prefix; do not apply this converter to an already-converted RobertaModel file.
This source witness is not a numeric state-dict/loading-equivalence certificate.

The supervised config's _name_or_path mentions bs512/lr1e-5; the unsupervised
one mentions bs128/gpu4/lr3e-5/l32. These are config literals, not proof of the
downstream recipe or a reason to overwrite CTL's batch512/lr1e-5/epochs10. The
official run examples initialize BERT-base, not these RoBERTa-large releases;
their seeds/lengths/optimizer or SimCSE objective must not become CTL defaults.

`check_retacred_ctl_drp_simcse.sh --describe` needs no backend/assets. Its audit
exits2 with `official_candidates_bound_paper_mapping_unconfirmed`; malformed
evidence exits1. Optional `--verify-cache PATH --tokenizer-fixture` runs only
tokenizers byte-BPE on synthetic strings. No train/model/weights/Git/network.
Both CTL rows and all existing strict/rank gates remain unchanged.


## Cited RECENT aggregation: recovered source, explicit CTL adaptation

CTL's bibliography identifies Lyu/Chen arXiv2105.08393. Its original paper
links [Saintfe/RECENT](https://github.com/Saintfe/RECENT/tree/4742c9e80517acbd1466a6fb3d5bcb4678e56c47).
Twelve public originals and six selected source blobs are locked. No raw task
dataset, example split or weights are fetched; no full upstream script runs.

The paper's operational algorithm groups learning samples by their exact ordered
entity-type pair and aggregates relation labels within each group. Both released
SpanBERT/GCN preprocessors contain the same r/s/o/t and set-aggregation AST.
Only that pure memory fragment is extracted and executed on synthetic fixtures.
This is not a per-relation Cartesian product of separately observed subject and
object types. Domain notation alone does not justify enlarging observed pairs.

The released preprocessing separately aggregates train/dev/test gold labels,
excludes no_relation rows from semantic groups, and for certain two-label groups
creates test semantic outputs directly from that split's relation set. The
evaluation path combines semantic outputs with a binary classifier. These
heldout-label-dependent operations are **not imported** into CTL. Observing these
source paths is not a measured estimate of leakage impact or a blanket verdict
on the original paper's results.

The released constant file has13multiclass type groups/33unique positive TACRED
labels. Six are outside canonical Re-TACRED;12canonical positive labels are
missing from these multiclass constants. Single-positive groups follow another
source branch, and old headquarters/parent/subsidiary/alternate-name labels
are not an interchangeable complete40-class Re-TACRED H table.

`fit_train_pairs(rows, source_split="train")` implements an isolated declared
train-only observed-pair primitive; runner-side physical split/hash binding is
still pending. It preserves exact type/label strings, rejects heldout fitting,
retains all training rows, adds NA unconditionally, and orders candidates by
the canonical40 IDs. These CTL/project boundaries are explicit adaptations;
the RECENT binary gate, direct-label shortcut and fixed TACRED maps are absent.

`candidate_labels(table, subject_type, object_type)` takes no gold label.
Known NA-only pairs remain valid. An unseen pair raises a diagnostic instead
of inventing all classes/NA-only/alias repair. Separately, coverage_diagnostic
reports unseen-pair/excluded-gold indices, never repairs candidates or skips a
sample. A one-positive known pair still competes with NA using CTL Eq2; it is
not a forced positive prediction. Tables carry semantic support hashes/counts
and explicit false physical-identity/exact-author-H/formal-admission flags.

The exact CTL author's H table/producer remains unconfirmed; citing RECENT is
not proof that CTL copied every source branch. Do not promote this primitive to
paper-faithful reconstruction without an explicit assumption review/sensitivity
plan plus exact checkpoint, recipe, scorer and neural/lifecycle evidence.

Portable check_retacred_ctl_drp_candidates.sh verifies source provenance and
the actual aggregation fragment offline; pending exit2, malformed evidence1,
descriptor0. No model training, checkpoint, dependency install or network.
Both CTL rows/default-five remain unchanged. Next: freeze a single explicit
paper-interpretive reconstruction assumptions/gates record before neural coding.


## Explicit reconstruction assumptions and acceptance matrix

See [the reconstruction matrix](ctl-drp-reconstruction.md) and its dedicated
`ctl_drp_reconstruction.json` design record. They distinguish recovered facts,
unimplemented proposals, sensitivity tests and pending acceptance evidence.
Their status is assumptions_prepared_neural_pending, not a runnable/certified
baseline. The current evidence grade remains not_verifiable; a future runnable
reconstruction starts as paper_interpretive. Exact author checkpoint/H and
paper scorer binding remain unconfirmed. Ready-five and both blocked CTL rows
are unchanged. No existing runner reads this record as a configuration/admission
override. Next is synthetic tiny joint-encoder/Eq2 FP32 testing, not full-data
training, weight loading or publication; independent review remains pending.


## Actual tiny neural core and owner-approved reconstruction policy

The original design matrix is a dated pre-implementation snapshot. Current
choices are in ctl_drp_runtime_choices.json; actual code is ctl_drp_neural.py
and ctl_drp_smoke.py. Owner permits reasonable declared gap filling. Exact
author checkpoint/H identity is not a prerequisite for an explicitly labeled
paper-interpretive reference, but it remains unconfirmed, never renamed exact.

The marker placement is now directly bound to the native example: [u1] subject
name (subject type) [u2], analogously [u3]/[u4] for object. Lowercase type text,
underscore-to-space and sentence-pair joining are explicit rendering choices.
Raw role/span/type metadata is retained. Inference encoding never reads gold.

Injected RoBERTa jointly encodes each candidate. Raw first final-layer token
feeds one shared affine scalar head; each sentence owns its Eq2 denominator.
Loss is mean NLL across sentences, not candidate-row-weighted or in-batch CE.
Known NA-only has p1/loss0; malformed grouping/overflow/missing gold fail without
candidate repair or sample skipping. Tiny initialization is random, not SimCSE.

Synthetic train/test traces contain actual word embeddings, Q/K/V, attention
scores/probabilities, intermediate layer/head I/O, candidate probabilities,
gold/prediction/loss and training parameter gradients. CPU FP32 post-update
model/optimizer/RNG/cursor recovery is a diagnostic, not CUDA/topology proof.
Core tests and canary evidence do not imply full-task data qualification,
large checkpoint loading, full runner integration or formal score reproduction.
