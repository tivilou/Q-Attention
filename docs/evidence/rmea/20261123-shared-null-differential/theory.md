# Shared-null differential readout: exploratory task hypothesis

The demonstrated problem is explicit ordered-entity-pair relations missed as
no_relation. Same-cache historical strong heads share 1155 positive-to-null
errors; fixed TRAIN semantics support five of eight sampled misses. Neither
statistic establishes that quantum interference caused or can repair the errors.
The previous frozen competition probe repaired none of eight positive misses.

For class r and token t let alpha_rt be the original row-normalized attention
and u_rt the original class utility. The proposed positive-class margin is

    m_r = b_r-b_0 + sum_t (alpha_rt-alpha_0t)(u_rt-u_0t),  m_0=0.

It trains the original relation effects and the separately learned shared null
effect jointly. All 40 classes, including null, remain in cross-entropy. There
are no gold labels, semantic cues, entity types or case strata in forward.
The null margin being zero is a softmax coordinate choice; it is not forcing
null probability to zero. The native control retains its exact original logits.

The simpler same-token pooled contrast sum alpha_rt(u_rt-u_0t) is contained in
the original model by replacing decoder_r with decoder_r-decoder_0 and setting
the null decoder to zero. We reject a new-expressivity claim for that variant.
The present two-distribution cross-difference instead changes the pooling rule.
No claim is made that the entire resulting model family exceeds the original.

Known identity: sum_t(alpha_rt-alpha_0t)=0. Therefore the readout rejects a
token-constant utility offset. It also rejects class-common utilities because
u_rt-u_0t is used. With v_t=u_rt-u_0t,

    |m_r-(b_r-b_0)| <= TV(alpha_r,alpha_0) [max_t v_t-min_t v_t].

This is the standard total-variation expectation bound, not a new theorem.
It predicts a failure mode: identical attention distributions, single-token
inputs or token-constant relative utilities leave only bias. Thus value.bias
and a class-common decoder component are ineffective directions in the signed
arms, unlike native. Parameter counts alone do not match effective capacity.
Foreground concentration alone has no guaranteed effect on the margin.

Nearest read original: Differential Transformer, arXiv:2410.05258v2, PDF pages
1,3,4 only, https://arxiv.org/pdf/2410.05258. Page3/Fig2 uses two softmax maps
subtracted with a learned lambda and shared value, followed by normalization.
This experiment fixes coefficient one, shares a task-specific null reference
across relations and uses class-relative utilities. The subtraction principle
is known; substantive novelty, quantum necessity and L5 are unresolved.
Unlike the closed 20261114 outcome-contrast probe, the two distributions here
come from different learned effects, not p and 1-p of the same measurement.
Unlike R102 decoder centering, this is a jointly trained change to pooling.
These distinctions license an experiment, not a novelty or superiority claim.

One prospective bounded TRAIN planning screen: fixed old4192 fit/4000
sentence-group-disjoint assessment, seed13, 2048 updates/arm, batch32,
unchanged AdamW/microbatch core, final update only. Coherent/dephased/native
share all original state; classical differential softmax shares the same
value/decoder/bias and order but has a different smaller router. All arms
learn from scratch. No assessment until all four final states exist. No VALID
or TEST reads. No selection among checkpoints or rescue by tuning budget.
Ordinary utility and dephasing attribution gates are reported separately.
If exact utility/problem gates fail, stop this profile; a short miss is not
all-family impossibility. A positive screen would still require full-data
independent utility and later same-core theory/novelty evidence for L5.
