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
