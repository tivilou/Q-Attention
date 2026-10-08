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
| MetaEntail-RE | [author paper](https://aclanthology.org/2025.naacl-long.165.pdf), Sections4/5.3 and Tables4/13; [official MNLI config](https://huggingface.co/FacebookAI/roberta-large-mnli/blob/2a8f12d27941090092df78e4ba6f0928eb5eac98/config.json) | Paper-derived40 templates, meta-class matrix, train-only types and entail grouping are primitives. MNLI revision, six small files, NLI order,512-token pair boundary and explicit expanded-pair budget are prepared; weight hashes are provider metadata only. General-domain recipe, actual head/runtime/lifecycle and abstention denominator remain unbound. Reported94.3 stays rank-ineligible; see `metaentail-contract.md` and `metaentail-neural-contract.md`. No ordinary-classifier substitute. |
| Curriculum Learning | [paper v2](https://arxiv.org/pdf/2107.09332v2);16 bounded public requests, two candidate trees | Paper method includes dependency GAT + bilinear classifier and reviewers trained on their own disjoint subsets. Table2 own baseline91.2 -> curriculum91.4 (+0.2pp). Candidate handle identity unverified and inspected entries describe different methods. Fold-count ambiguity, graph equations, pacing, exact recipe and scorer remain unresolved. See `curriculum-contract.md`; blocked, no marker-only substitute. |
| LUKE + CoRE | `vanoracai/CoRE@c98d6947`; paper Tables2/3 | Released code selects gold-conditioned entity-bias challenge examples and runs macro scorer on a TACRED-finetuned checkpoint; not full Re-TACRED micro reproduction. Cached preprocessing producer/full training/dev selection missing. See `luke-variants-contract.md`. |
| LUKE + CFIE / resampling / focal | Official `HeyyyyyyG/CFIE` **RE** branch@17583143; original paper author link verified | Default branch is NER; RE source uses NYT24 GCN/BERT, branch-specific losses/interventions. No verified LUKE/Re-TACRED port. Sampler existence is not the CoRE resampling recipe; focal loss fetch failed. Tables3/2 bind micro/macro and five-run medians, not project outputs. |
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

## LUKE variant score/source binding

- CoRE paper Table3 micro: unmodified LUKE90.3, resample90.5, focal90.4,
  CFIE90.5, CoRE90.9; Table2 macro:80.2/80.5/80.3/80.4/81.6. Reported
  aggregation is median of five training runs, not a single seed or mean±SD.
- Source `challenge_set` depends on gold labels; released `getF1` is macro
  over the gold numeric range with first entry excluded and absent-class P/R1.
  Small actual AST-slice fixtures demonstrate different denominators and
  source1/3 macro versus4/7 benchmark micro. These are fixtures, not project
  measurements or a claim that the paper's intended full-test result is invalid.
- Original CFIE author repository/RE commit recovered; twelve selected files
  hash-locked and never fully imported. Branch-specific loss and intervention
  expressions require source/paper reconciliation before any LUKE adaptation.
- Official LUKE training README/base configuration separately discovered at
  studio-ousia/luke@25165454. Its TACRED route is not the missing CoRE Re-TACRED
  port. Next gate is reader/model/attention and original checkpoint identity.
-24 paced HTTP requests,22 successful, two TLS failures retained without retry.
  No raw dataset or pretrained weights downloaded. Offline audit
  `python -m baselines.retacred.luke_audit` exits2/not_verifiable/zero jobs;
  optional `--verify-cache <cache>` verifies public bytes, not model fidelity.

## LUKE foundation prepared — 2026-10-08

Official reader/model/entity-aware attention and15 selected files locked at
251654548a15b9a8ddc708a53b715bdf3a112102. Pretrained luke-large identity at
0729d044dfe301d9ecabc222d60633f92ac450eb, provider SHA/size recorded for all
eight required assets;2.246GB weights not downloaded. Prepared wrapper executes
the actual parser and author HF classifier; uses inclusive role-ordered entity
positions and40-class biased head, not CLS. Scalar/vector fixture checks do not
certify BPE/backbone/autograd/weights/training lifecycle. All four LUKE comparison
ports remain blocked. No new cohort row or score is synthesized.
