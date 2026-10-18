# Q-ECPA toy implementation review

Scope: bounded CPU synthetic toy/preflight only. No formal experiment readiness
or task-benefit endorsement. Review performed before experiment execution.

Capability ranking used the dated 2026-10-17 ai-model-world inventory followed
by same-source BenchLM category evidence: Coding .5, Reasoning .3, Knowledge .2.
This is an explicit heuristic, not a measured review-quality benchmark.
Opus5.5 ranked first; its requested native max-effort mapping was unverified.
Compatible GPT6-Astra timed out after one request; next compatible Opus5 returned
503 after one request. No scientific opinion is attributed to those failures.
Bounded independent subagent fallback reviewed the same source packet.

Verdict: PASS after resolving two findings:

1. Replace automatic stage chaining with actual consumed references. Context now
   names values, scores, input embedding and mask; pooling names hidden states
   and entity/sentence masks. Diagnosis has real numeric values.
2. Bind each captured state to actual optimizer update and deterministic model
   parameter/buffer digest; reject resumed trace source/config/model mismatch.

The reviewer found no remaining must-fix defect in Hermiticity, probability
orientation, same-H dephasing, gradient paths, padding masks, target boundaries,
matched parameter budgets or the scoped restart lifecycle.

Reviewed executable hashes:

| File | SHA-256 |
| --- | --- |
| plugins/q_ecpa.py | e04a8d3321d5d707dd67486417a507a9e988b2457d2460eeffe09d1d4487add1 |
| adapters/q_ecpa.py | 8222c9fd66a79fdc54369d21388aa2feba35e98326fa9fb250ea04d30a91c055 |
| analysis/ecpa_trace.py | d255b8e5a99014dbeb303b96dfb70d61727696c7375cbc8f00c6d4317dd4cadb |
| tasks/ecpa_toy_data.py | 8f35d4383d03318583c81707316329b55e7fb5fd9c487d520949fe038443ab85 |
| experiments/run_ecpa_toy.py | 7797894c7d8e07a9b67f16a426e371d8f0100350c25639676ebac2007e4a613d |
| scripts/run_ecpa_toy.sh | 95397e2523f9cd33882942d966ce2e36a573a45f73c15ebedbbfbe6722806c3f |
| tests/test_ecpa.py | 2aa723d35739333218de42e2c7262346fa98813d927fb73bc463e302cd072769 |

Execution still has to verify tests, actual launcher, interrupted restart,
shared structural validation and all five bounded conditions.
Formal follow-up additionally needs original-QK held-out training comparison,
stronger entity-permutation control, GPU/formal-data integration, and actual
portal tensor projection inspection. A constant-H unitary repeated m times is
equivalent to evolution for m times the duration; m is not independent circuit
expressivity. Q versus dephased still tests cross-step coherence removal.

Host follow-up: cache only the parsed trace metadata by resolved path, nanosecond
mtime and byte length (bounded to eight documents). Value artifacts still receive
hash verification on every read. This removes repeated large JSON parsing during
the exhaustive coordinate gate; numerical/readout semantics are unchanged.
Focused execution and corruption/restart tests cover the follow-up revision.
