# Same-checkpoint gate readout diagnostic

Freeze the validation-selected quantum_corroboration checkpoint from the completed
gen88 full study. Use all58465 TRAIN rows in original order/native128-sized
batches, split whole batches across two allocated GPUs. No fitting, new seeds,
VALID/TEST evaluation, checkpoint writes or COS upload. Reuse unchanged plugin,
environment, cache, host wrapper and selection; do not rescue the closed profile.

Compare actual native gate-on forward against gate-off at the same key/query,
value/decoder/bias parameters. Gate-off is exactly (base+EPS)*mask followed by
the original normalization. It is not the separately trained gaussian_base.
Check actual native forward versus same-batch reconstructed gate-on logits and
argmax before interpreting any effects. No original TRAIN logits were stored;
this is forward/component consistency, not a claimed parent TRAIN-metric replay.

Keep actual FP32 logits for both scientific outcome tables. Independently check
the exact reweighting/covariance identity in FP64 from the actual FP32 base,
gate and value components. Explicitly distinguish this algebraic verification
from FP32 correspondence (scaled tolerance2e-5). Covariance identity/margin
identity absolute tolerance1e-10. Native reconstruction tolerance1e-5 absolute
and relative, exact argmax/CM. All values finite; checkpoints/inputs pinned.

For off baseline alpha and c=(b*g+EPS)/(b+EPS), delta logit is
Cov_alpha(c,u)/E_alpha(c). For strongest non-gold off-rival s0, exact
delta(gold-minus-max-rival) is delta_gold-delta_s0-switch_penalty,
where switch_penalty=max_non_gold(on_logits)-on_logits[s0]>=0.
Include this rival switching term rather than assuming one rival remains strongest.
Same bias cancels. Padding has zero mass in every formula.

Report all40 confusion matrices/Macro-F1/CE, all-row repair/harm and null
transitions, original-on-correct and original-on-wrong strata, all gold-v-max-rival
margin changes and decomposition residuals. Report gate statistics and softmax
loss but do not select a favorable class/metric/cohort. Positive/negative/null
effects are all valid diagnostic outcomes; no training promotion gate.

If numerical/source/replay checks fail, mark technical-invalid and do not interpret
downstream effects. No cohort/batch/tolerance/checkpoint rescue. This measures
fixed-parameter routing effect on exposed TRAIN, not training causality,
generalization, semantic binding, independent quantum contribution or novel theory.
Later method selection requires a structurally different falsifiable mechanism.

Private logits and per-row arrays stay on the GPU host. Public output is only
summary.json and RUN_COMPLETE plus source/design/review and aggregate analysis.
Checkpoint/cache are already durable dependencies; this disposable forward needs
no new COS object or lifecycle/resume matrix. Timeout600s,2GPU/4CPU/8GiB RAM.
