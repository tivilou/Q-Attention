# Source audit — 2026-10-07

Scope: recover the previously discussed fifteen reported configurations. This
is not evidence of a globally complete top-15 leaderboard. Paper ranks and
scores are historical selection aids, never project measurements.

| Family | Official artifact inspected | Concrete implementation boundary |
| --- | --- | --- |
| Improved Baseline | `wzhouad/RE_improved_baseline@fe00781116ab4c34cbcd9a2ae5b02513afae87ee` | Four active marker variants imported; `RETACREDProcessor` adds typed markers during tokenization, requiring resize **after** data encoding and **before** optimizer creation. Source mechanism tested on tiny RoBERTa architecture. Full checkpoint/GPU gate remains collaborator-side. |
| PTR | active code `thunlp/PTR@76f4685f4204f9d469ba1f82ce1dd02ad52aab42`; recovered metadata `thunlp/PTR@420926b83025f313f27d77dbb5c74d7089121b89` | `run_prompt.py` loss is **mean of five verbalizer CE objectives PLUS relation-level CE on the summed raw logits**; omitting the latter is not faithful. Two AdamW optimizers. The cloud share is expired; official history contains the hash-locked 40-row verbalizer and 40-class mapping. **Normal adapter implemented; full-weight/CUDA gates pending.** Reverse recipe remains unverified. `input_flags` are all zero in released tokenization; do not invent an active soft-prompt branch. See `ptr-mechanism-contract.md`. No fork raw train/test data substituted. |
| Sequence Routing | `bahramiramazan/re-representation@cc95c4ad20b9ed46b978ffa4e88e31ce28c7ed20` | `Experiments.py` imports missing `Abstraction`; `rc_util.py` opens missing `essential_files/Relation_counter_retacred.json`. `get_settings` declares `n_class=41`, `h3_embed_size=41`. Our canonical data has 40 classes. `RC_Model` depends on external Heinsen routing and all 25 hidden states. Exact published H3/Mix recipe and license must be recovered before an adapter can be labelled faithful. |
| CTL-DRP | `Viva28/CTl-DRP@2db67a258c266fba3ee0e2cbcdb88ccbc45cf40d` | Only `README.md` containing “Coming soon”; both reported supervised and unsupervised configurations remain blocked. |
| MetaEntail-RE | [author paper](https://aclanthology.org/2025.naacl-long.165.pdf), footnote 1 | Official link `https://github.com/wphogan/metaentail-re` returns 404; clone says repository not found. Exact hypotheses, meta-class graph, type-pair filtering and grouped selection cannot be replaced with an ordinary classifier. Evaluation convention needs checking before ranking against non-null benchmark F1. |
| Curriculum Learning | [author paper](https://arxiv.org/abs/2107.09332) | No verified executable official repository identified. Cross-review difficulty estimation and curriculum scheduling need a declared paper implementation contract, assumptions and independent mechanism review. |
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
