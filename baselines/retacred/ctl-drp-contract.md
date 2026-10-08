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
paper reconstruction is certified. Next: bind sup/unsup checkpoint identity and
explicit H provenance, then implement the candidate-conditioned encoder/lifecycle.
