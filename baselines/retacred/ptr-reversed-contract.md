# PTR Reversed: changed-task reference, not canonical-task baseline

The historical reported91.9 is retained, **not** certified same-task SOTA or a
project measurement. Normal PTR remains a separate implemented adapter.

## Primary evidence and recovered boundary

- [Paper v3, section4.2](https://arxiv.org/html/2105.11259v3#S4.SS2) defines
  reversal of only part of the relations. Table4 reports91.9 for ReTACRED;
  Table5 illustrates a reversed entity order.
- [Repository-member explanation](https://github.com/thunlp/PTR/issues/3#issuecomment-899019118)
  uses the inverse of `per:origin` with the entity pair exchanged.
- [Subset explanation](https://github.com/thunlp/PTR/issues/3#issuecomment-900008186)
  says `per:origin` is the main reversed relation, based on prompt suitability.
  **Mainly** is not a complete executable subset or exact inverse verbalizer.
- [Comparability explanation](https://github.com/thunlp/PTR/issues/3#issuecomment-900011357)
  explicitly separates this setting from the conventional task. The changed
  direction can distinguish origin from countries-of-residence; this is not
  merely a different neural layer for the same frozen input contract.
- Active source `thunlp/PTR@76f4685f4204f9d469ba1f82ce1dd02ad52aab42`
  has only the normal input path. Historical trees at420926b,0188c80,a9ba370
  contain the same normal ReTACRED verbalizer blob. The public default history
  has13 commits and one branch was returned. This bounded inspection is not
  proof about unpublished/private code or all uninspected objects.

## Scientific and implementation gates

1. Keep `ptr_reversed` outside `ready` and the canonical strict comparison set;
   `reported_score_rank_eligible=false`, `benchmark_comparable=false`.
2. Do not swap all entity pairs and call it the original partial reversal. Do
   not map inverses by guessing canonical relation names or reuse normal
   verbalizers as if they were the original reversed file.
3. A selective transform using the gold relation to choose entity order can
   make model input reveal target-dependent orientation. This is a **risk
   derived from the proposed transform**, not a claim of leakage in normal PTR.
   Changing gold alone must not change canonical-task model inputs. Type-only
   routing is not a certified replacement: origin and residence can share the
   same entity-type pair.
4. Any future changed-task reproduction needs its own qualified task/data
   contract, full inverse-label/verbalizer mapping, reversible output mapping,
   source or declared paper assumptions, and separate result table. It is not
   admitted by removing a guard from the common runner.
5. Paper budget is5epochs/batch64,10% warmup and best-development checkpoint.
   Released normal shell uses warmup500 and the runner evaluates final weights.
   The common project's final-epoch protocol is a declared adaptation, not
   the published reversed recipe.

## Read-only audit

`python -m baselines.retacred.ptr_reversed_audit` verifies the pinned normal
source and prints the recovered evidence and unresolved gates. Expected exit2,
`not_verifiable`, zero model jobs. No source imports, weights, network or
training. `ptr_reversed_evidence.lock.json` binds source hashes, paper identity
and the member-comment provenance; it is an audit lock, not a fidelity certificate.
