# Curriculum Learning: paper mechanism and unresolved recipe

**Admission: blocked / not_verifiable; no model adapter.** This is an audit of
[Park and Kim, arXiv:2107.09332v2](https://arxiv.org/pdf/2107.09332v2),
not a reproduction of its published result. The evidence lock records public
response hashes; original PDFs and downloaded source stay in private caches.

## Paper facts (sections 2.1, 2.2, 3.1 and Table 2)

- RoBERTa-large produces 1024-dimensional encodings. Entity marking uses
  punctuation symbols from Zhou and Chen; symbol states encode the entities.
  The table compares against typed markers, but the exact inserted strings and
  token/span alignment are not specified in this paper.
- A **dependency-graph GAT** encodes syntactic information. Its output is
  concatenated with entity encodings and fed to a **bilinear relation classifier**.
  Removing GAT or replacing the bilinear classifier with a generic MLP changes
  the method behind the reported score.
- Cross review partitions the corpus without replacement. Each independent
  reviewer trains on **its own subset** and predicts the other subsets. More
  incorrect reviewer predictions indicate harder/noisier examples. This is
  not ordinary out-of-fold training on all the other folds, and is not sorting
  by one model's training loss. The target model gradually learns difficulty
  groups and ultimately the entire corpus.
- Section 3.1 gives AdamW, learning rate **0.0005**, layer decay **0.7**, dropout
  **0.1** on Transformer/linear layers. These are literal reported values,
  not the marker adapter's current hyperparameters.

Table 2, Re-TACRED, as reported (not project measurements):

| Configuration | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Authors' typed-marker reimplementation | 90.6 | 91.3 | 90.9 |
| Authors' GAT baseline | 91.7 | 90.7 | 91.2 |
| Authors' baseline + curriculum | 91.3 | 91.5 | 91.4 |

Thus the curriculum increment over **its own baseline** is **0.2 percentage
points**. The cited typed-marker row is91.1. Neither number is a measurement
of our project baseline, and the paper does not specify a scorer implementation
or null-label handling, so evaluator equivalence remains unverified.

## Source search: bounded findings, not proof of absence

Sixteen paced public HTTP requests, zero retries/browser escalations. Search
covered the title/id, author-related repositories, arXiv and the cited ACL
cross-review antecedent. One guessed lab query returned422; it contributes no
absence evidence. An account matching the public author identifier has19
public repositories, but its profile supplies no verified paper-author link.
Keep its identity **unverified**; do not label it an official implementation.

Two complete candidate trees and selected blobs were inspected without
executing them:

- `a163912/RE_MTB` README cites **Matching the Blanks** and TACRED, not this
  curriculum paper. Its selected training entry is ordinary shuffled-data
  training with Adam and cosine scheduling; it is not evidence for the
  paper's AdamW/layer-decay/cross-review recipe. The inspected tree does not
  include the imported `models.Model` definition.
- `a163912/RelationExtraction` README describes Korean relation-extraction
  context-masking multitask learning and soft-voting ensembles. The tree and
  README are not a verified implementation of the curriculum method.

The cited [ACL 2020 cross-review paper](https://aclanthology.org/2020.acl-main.542/)
is a different paper; its defaults must not silently become the Re-TACRED
training recipe. No private history, uninspected branches or unavailable
supplements are claimed absent.

## Material gaps and reconstruction boundary

Before a runnable reconstruction is proposed, explicitly resolve or declare:

1. Exact punctuation/type markers, subword/word dependency alignment and parser.
2. GAT topology, edge types/directions/self loops, heads, depth, dimensions and
   graph readout; concatenation operands and bilinear classifier equation.
3. Cross-review fold count and balancing: section2.2 indexes0..n but says n-1
   predictions per example. The numerical count is not silently corrected.
4. Reviewer initialization/budgets/seeds and the exact correctness aggregation.
5. Difficulty bucket boundaries, easy-to-hard pacing, prior-group replay,
   stage update/epoch counts, and optimizer reset/continuation semantics.
6. Batch size, total epochs, maximum length, loss, layer-wise LR-group mapping,
   weight decay, Adam epsilon/betas, warmup/scheduler, precision and clipping.
7. Label mapping, split identities, seeds, checkpoint selection and evaluator.

Allowed future project boundary: canonical row conversion, hash-locked split
identity and complete common metrics, separately declared from published-score
replay. Forbidden substitutions: marker-only classifier, textbook GAT defaults
called original, same-model loss sorting, standard leave-one-fold-out reviewers,
or importing an unrelated candidate's optimizer/budgets.

Any reconstruction with material choices stays `paper_interpretive` and outside
the strict set. Paper-faithful admission requires equation/toy tests, sensitivity
checks, independent mechanism review and declared-budget empirical evidence.
Train/test case studies must capture actual dependency graphs, embeddings,
GAT layer/attention tensors, entity pooling, bilinear scores and gradients;
cross-review must record reviewer identities, training membership and votes.

## Offline check

```bash
python -m baselines.retacred.curriculum_audit
python -m baselines.retacred.curriculum_audit --verify-cache <public-evidence-cache>
```

Expected exit2: `not_verifiable`, zero model jobs, no downloads or upstream
execution. The optional cache check verifies every accepted public response
hash; it does not certify model fidelity. A missing/tampered cache exits1.
`--models curriculum` and `--models all` are rejected before worker launch;
the default five admitted adapters are unchanged.
