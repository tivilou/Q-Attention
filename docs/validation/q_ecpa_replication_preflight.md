# Q-ECPA replication preflight

Only new seeds29/53 train; audited seed13 is imported by immutable Git hashes. Scientific source remains unchanged. The global queue has16 jobs, each seed owns its baseline and seven conditions. Full-data runs remain collaborator-only.

Seven focused tests passed on both development and GPU hosts; three pure statistics checks also passed locally. The actual CUDA parent passed2GPU pause with injected OOM,1GPU same-run completion and completed2GPU no-op. The quantum execution tier persisted, all owned children exited, and complete-run checkpoint hashes stayed unchanged.

Both new seeds produced complete full numeric cases, with every NPY validated. Injected export failure left no final/partial report; retry from the same immutable run and identical re-export passed. Numeric corruption was rejected and original bytes restored. A failed synthetic baseline blocked only its own selectors while the other seed completed. A disposable bare Git remote passed report-only commit/push, forced push failure, retry without a duplicate commit, and rejection of unrelated dirty files.

Private numeric transport in this new canary is a synthetic stub; production HTTPS uses the previously tested unchanged writer with unique per-seed IDs. Synthetic metrics are not formal utility evidence. Exact hashes and timings are in the JSON companion.
