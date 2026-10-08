# 2025 RoBERTa comparison: paper evidence, not a recovered implementation

## Identity and published result

The reference is *Transformer Enhanced Relation Classification: A Comparative
Analysis of Contextuality, Data Efficiency and Sequence Complexity*, Bowen Jing,
Yang Cui and Tianpeng Huang, [arXiv v1](https://arxiv.org/abs/2509.11374v1).
Table1 binds the RoBERTa row to micro-F1 percentages74.84/83.09/91.53 for
TACRED/TACREV/Re-TACRED. Its caption separates TACRED training with TACRED or
TACREV evaluation from Re-TACRED training and evaluation. Appendix Table2 lists
Re-TACRED58465/19584/13418 train/dev/test examples and40 classes. These are paper
facts, not our data identity certificate, task qualification or measurements.

## Mechanism and adaptation boundary

Section3.2 describes entity-preserving typed punctuation input: subject markers
use @ and type delimiters *, object markers use # and type delimiters ^.
This overlaps the existing Improved adapter's input family. That fact does not
establish identical type normalization, byte-level BPE, marker positioning,
pooling or classifier. The paper names RoBERTa but does not identify base/large,
checkpoint revision, head or forward/loss implementation. Do not route this row
through the existing typed_marker_punct adapter as an author reproduction.

The lowercasing, entity replacement and GloVe vocabulary discussion belongs to
the non-transformer subsection. Do not silently apply it to the RoBERTa path.

## Missing recipe and scorer

Section3.1 states that a grid search was used. No grid or selected numeric
hyperparameters are supplied in the inspected v1 PDF/HTML/TeX. No author source
link was identified in those materials, and two exact public repository queries
returned zero hits. This bounded observation is not a global source-absence
claim, not evidence of a deleted/private repository, and not a checkpoint lock.

Required unresolved choices: checkpoint/tokenizer identity, type normalization,
pooling/head/loss, optimizer/LR/scheduler/warmup/dropout, batch/accumulation/epochs,
seed set and aggregation, grid/checkpoint selection split and rule, length and
overflow policy, label order, scorer/no_relation denominator. The paper's
micro-F1 label alone does not bind the project's non-null micro convention.

## Evidence and promotion

`marker_2025_evidence.lock.json` pins six public original responses, exact query
scope, native TeX identity, table values and unresolved choices. Raw bytes are
private `.external` evidence, never imported as model code or copied into Git.

`bash scripts/check_retacred_marker_2025.sh --verify-cache CACHE` verifies the
original hashes, receipt scope, actual TeX table and input statement, public
query results and linked cohort state offline. It exits2 with
paper_recipe_unverified, zero model jobs and formal admission false. Invalid or
tampered evidence exits1. Without a cache, the declared lock can be inspected,
but actual originals are explicitly not verified. `--describe` exits0 and
requires neither Python nor assets. No command trains, downloads, changes Git,
loads a checkpoint or fills missing recipe choices.

Retain all fifteen historical rows plus the classic anchor. Ready adapters stay
five and the historical91.53 is rank-ineligible pending source/recipe/scorer and
runtime gates. A later explicit common-protocol reconstruction must remain
paper_interpretive; it is not this paper's score reproduction or a new family.
