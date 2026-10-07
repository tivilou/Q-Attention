# Source audit — 2026-10-07

Scope: recover the previously discussed fifteen reported configurations. This
is not evidence of a globally complete top-15 leaderboard. Paper ranks and
scores are historical selection aids, never project measurements.

| Family | Official artifact inspected | Concrete implementation boundary |
| --- | --- | --- |
| Improved Baseline | `wzhouad/RE_improved_baseline@fe00781116ab4c34cbcd9a2ae5b02513afae87ee` | Four active marker variants imported; `RETACREDProcessor` adds typed markers during tokenization, requiring resize **after** data encoding and **before** optimizer creation. Source mechanism tested on tiny RoBERTa architecture. Full checkpoint/GPU gate remains collaborator-side. |
| PTR | active code `thunlp/PTR@76f4685f4204f9d469ba1f82ce1dd02ad52aab42`; recovered metadata `thunlp/PTR@420926b83025f313f27d77dbb5c74d7089121b89` | `run_prompt.py` loss is **mean of five verbalizer CE objectives PLUS relation-level CE on the summed raw logits**; omitting the latter is not faithful. Two AdamW optimizers. The cloud share is expired; official history contains the hash-locked 40-row verbalizer and 40-class mapping. **Normal adapter implemented; full-weight/CUDA gates pending.** Reversed setting is a selectively reoriented task; author explicitly marks it non-comparable with conventional setting. Complete inverse subset/verbalizers remain unverified; see `ptr-reversed-contract.md`. `input_flags` are all zero in released tokenization; do not invent an active soft-prompt branch. See `ptr-mechanism-contract.md`. No fork raw train/test data substituted. |
| Sequence Routing | `bahramiramazan/re-representation@cc95c4ad20b9ed46b978ffa4e88e31ce28c7ed20`; historical `Abstraction.py@6065881f` recovered | Active experiment4 is H3/Mix, **H3 output512 overrides the initial41**, classifier remains41 vs canonical40. Missing metadata is generated later, but factory reads it before generation. Source budget24/5 differs from paper64/6. Released evaluator includes null labels; all-label micro-F1 equals accuracy. The published92.2 binding is unverified, not certified same-metric ranking. Dependency experiment revision/main license unresolved. See `sequence-routing-contract.md`; row remains blocked, no source patch/model substitution. |
| CTL-DRP | `Viva28/CTl-DRP@2db67a258c266fba3ee0e2cbcdb88ccbc45cf40d` | Only `README.md` containing “Coming soon”; both reported supervised and unsupervised configurations remain blocked. |
| MetaEntail-RE | [author paper](https://aclanthology.org/2025.naacl-long.165.pdf), footnote 1 | Official link `https://github.com/wphogan/metaentail-re` returns 404; clone says repository not found. Exact hypotheses, meta-class graph, type-pair filtering and grouped selection cannot be replaced with an ordinary classifier. Evaluation convention needs checking before ranking against non-null benchmark F1. |
| Curriculum Learning | [paper v2](https://arxiv.org/pdf/2107.09332v2);16 bounded public requests, two candidate trees | Paper method includes dependency GAT + bilinear classifier and reviewers trained on their own disjoint subsets. Table2 own baseline91.2 -> curriculum91.4 (+0.2pp). Candidate handle identity unverified and inspected entries describe different methods. Fold-count ambiguity, graph equations, pacing, exact recipe and scorer remain unresolved. See `curriculum-contract.md`; blocked, no marker-only substitute. |
| LUKE + CoRE | `vanoracai/CoRE@c98d69472110d2514458e3b58a6a82f9793974aa` | Released `core.py` directly runs `studio-ousia/luke-large-finetuned-tacred` on repository TACRED `test.json`, with `preprocess.data` precomputed scores/constraints. It does not provide the Re-TACRED train/final-selection pipeline behind the reported row. Precomputed test assets are not substituted for project data. |
| LUKE + CFIE / resampling / focal | comparison rows in [CoRE paper](https://arxiv.org/abs/2205.03784) | Not three implementations in CoRE. Original CFIE, sampling distribution and focal alpha/gamma, backbone/initialization and budgets remain unpinned. Arbitrary textbook losses would be interpretive controls, not certified reproductions. |
| 2025 RoBERTa marker comparison | [paper](https://arxiv.org/abs/2509.11374) | Same architecture family as Improved; separate recipe not locked. Keep duplicate configuration visible without claiming another independent model. |

## Runtime findings repaired in the new adapter

- Typed special markers enter the tokenizer during `encode`; resizing the
  embedding table in the constructor caused an out-of-range lookup. The
  behavioral gate caught it; finalization now precedes optimizer registration.
- A tensor backward hook must return `None`; returning a metadata string would
  replace the gradient and break backward. Tracing transparency is numerically
  tested against the uninstrumented upstream model.
- Upstream-import cache files must not cause false “dirty source” failures.
  Imports now compile pinned `.py` directly and never execute a stale `.pyc`.
- Exchange helper receipts use `status=uploaded`, not `complete`. The lifecycle
  validates all receipt flags before projecting the delivery status to complete.
- Public export does not copy private sample JSON or numeric tensors. Its
  compact exchange mapping must point to the actual numeric artifacts.

## Promotion boundary

The default suite contains four marker configurations and normal PTR. Normal
PTR passed tiny CPU source, numeric trace and dual-state recovery gates. Their
checked-in manifests remain `not_verifiable` for full-weight formal
certification; tiny source fixtures are not promoted into CUDA/AMP evidence.
The complete cohort remains unfinished. `--models all` rejects admission
before any worker starts. Normal PTR imports the five verbalizer objectives
plus relation CE, retains both optimizer/scheduler states, FP32 and zero flags,
and records actual masked-state projections. Sequence Routing recovery remains separate.
A paper reconstruction must declare assumptions and evidence limits before
entering a separate run set.

## Sequence Routing recovery and metric correction

- Official historical tree recovered `Abstraction.py`; its locked SHA256 is
  `df4c064ee65ba840e5edbb91df89a4dc0672a990e3eaf8cd8d1e0869767d82dc`.
  It stays outside the active checkout and public Git, and is never executed.
- Bounded official/public inspection found29 default-branch commits, inspected
  16 historical trees, one branch, no tags/forks. Histories for the two exact
  missing metadata paths and root LICENSE were empty. This is a bounded finding,
  not proof of absence from private or uninspected refs.
- `rc_data` can generate relation files but has a clean-checkout bootstrap
  ordering problem. First-occurrence label/token order and41-output policy have
  not been certified; no fabricated metadata is promoted as original.
- Actual source metric assignment on a fixed fixture gives .9 all-label
  micro-F1/accuracy versus0 benchmark non-null micro-F1. Historical ranks/scores
  stay visible, with Sequence Routing explicitly ineligible for same-metric rank.
- Recovery CLI: `python -m baselines.retacred.sequence_resources`;
  offline cache validation: add `--verify-only`. Offline readiness audit:
  `python -m baselines.retacred.sequence_audit`, expected exit2/not_verifiable.
  Neither imports upstream code nor launches training; source pin remains clean.

## PTR recovery provenance

- Bounded public search: expired cloud share plus 25 public fork trees,
  27 requests, no retries/browser or access-control bypass. Multiple fork trees
  contain verbalizer blob `495dbf5a5eb04e61d4fe7f1b46d6cc1120bd8d4a`.
- [Official historical commit](https://github.com/thunlp/PTR/commit/420926b83025f313f27d77dbb5c74d7089121b89)
  remains accessible; exact metadata files are downloadable from its
  `datasets/retacred/` tree, not merely a third-party fork.
- Verbalizer SHA-256 `70f57788b93a83ea968c9ede19b79391a138b230e954ed01433deeb10f9799f6`;
  mapping SHA-256 `bedf6ac6216a4098a0d174c610dec086d59ae455b5fb81dbd48a8d16ce83ce89`.
- Recovery CLI: `python -m baselines.retacred.ptr_resources`; validate-only:
  `python -m baselines.retacred.ptr_resources --verify-only`.
- This corrects the earlier resource-unavailable finding. The normal adapter
  is now implemented; full-weight CUDA gates are still pending and this does
  not define or certify the reversed variant.

## PTR Reversed comparability correction

- [Repository-member clarification](https://github.com/thunlp/PTR/issues/3#issuecomment-900011357) separates the reoriented task from conventional input. The main reversed relation is `per:origin`, not a verified complete subset.
- Paper v3 section4.2 and Table5 alter entity/relation direction. Active source only implements normal order; three inspected historical trees share the same normal metadata blob. No raw train/test files were fetched.
- Preserve historical91.9 and its row, but set same-task rank/comparability false; `ready` remains5. No global entity swap, guessed inverse mapping or normal-PTR fallback is admitted.
- Offline audit: `python -m baselines.retacred.ptr_reversed_audit`, expected exit2 with zero model jobs. Gold-conditioned orientation is a target-input risk to test in any future changed-task contract, not evidence of leakage in normal PTR.

## Curriculum Learning mechanism and source audit

- The reported91.4 binds to RoBERTa entity encodings **plus dependency GAT and
  a bilinear classifier**. Its own GAT baseline reports91.2. Do not substitute
  the existing marker MLP or claim +0.5 over its typed-marker reimplementation
  is the curriculum's isolated gain.
- Cross-review trains each reviewer on its own independent subset and reviews
  the others. Paper reviewer-count notation is inconsistent; exact groups,
  pacing and reviewer budgets remain unspecified. Ordinary OOF/loss sorting
  and antecedent-paper defaults are not recovered Re-TACRED facts.
- Sixteen paced public requests, one invalid422 query, no retries/browser.
  Candidate account identity is unverified; inspected RE_MTB/RelationExtraction
  entries describe different approaches. This is not a global source-absence
  claim. Original public bytes stay outside Git; hashes are locked.
- `python -m baselines.retacred.curriculum_audit` exits2/not_verifiable, zero
  jobs. Add `--verify-cache <cache>` to check all fifteen successful public
  response hashes without fetching or importing downloaded code. Rank/evaluator
  eligibility remains unverified; defaults remain five adapters.
