# Fixed native readout / Balanced Softmax objective screen

Question: does known class-prior-aware training improve the native quantum
router's all40 Macro enough to survive matched controls? Gen93's three folded
models lowered average assessment CE but lost Macro, with more positive-to-null
errors. This motivates a test, not a diagnosed imbalance/quantum cause.

Use the original factorized value/decoder, native quantum and native dephased
attention, and the exact gen85 anisotropic Gaussian base. All three get the
SAME training-only log count correction. No folded gen93 model, common frozen
utility, readout expansion, sampling change, additional features or mixture.

For raw logits z and FIT-only counts n, minimize CE(z+log(n), y). At inference
return z exactly, with no threshold tuning. This is known Balanced Softmax,
verified against the authors' README and loss implementation, not original
theory, not meta-sampling, and not a guarantee of optimizing Macro-F1.
Train/test prior-shift assumptions and class-balanced accuracy are distinct
from the actual empirical Macro-F1 gate. All40 counts are positive; no smoothing
or strength parameter. Counts come ONLY from original fixed8192 fit indices:
cohort8fb8393b, not assessment/VALID/TEST. Freeze their entire40-vector in config
and model buffer; verify against the actual private cohort before staging.

Fixed original TRAIN8192 fit/4096 assessment, sentence-group disjoint, all40;
seed13,1024updates,b128,micro4,FP32/complex64,TF32off, originalAdamW lr.001,
wd.01,betas.9/.999,eps1e-8,clip1. Three new fits finish BEFORE final1024
assessment. Train losses now mean Balanced Softmax CE; assessment CE uses raw
logits and is comparable across the reused endpoints. Save all40 confusion,
support,F1,CE and full learning history, not only rare or positive classes.

Reuse nativeQ/D gen90, Gaussian gen85 and Gaussian covariance gen91, with exact
same cohort/order/native initialization method/1024 budget; no refits. Gaussian
covariance is a stronger practical function-class reference, not a parameter-
matched loss ablation. NativeQ/D/G all81028 trainable scalars; count buffers
do not add parameters. Do not assert equal effective capacity or quantum speedup.

Primary promotion: adjustedQ strictly beats adjustedD/G and all four reused
references on all40 Macro, six fixed gates. An improvement against nativeQ
alone tests utility but does not establish quantum-specific contribution.
Pass permits full single-seed protocol design ONLY, no automatic full/multiseed.
If any gate fails, defer exact full-strength/1024 prior-adjusted profile; no
prior strength, class-weight, threshold, LR, seed, cohort, budget or control
rescue. Small-screen failure is not family impossibility. Check direction of
positive-to-null and null-FP descriptively, without replacing primary Macro.

Use unchanged screen50ce858f/host238f7e68, fresh shared twoGPU allocation,
3072physicalupdates,900sec/arm/3600sec campaign. Review only affected objective
semantics and actual8row updates; no core resume/COS lifecycle matrix. Short
restartable screen keeps code/config/compact reports in Git and localbest/latest,
zero new COS. Existing durable inputs remain protected. Windows receives no
task arrays/checkpoints/features. L3/L4-T NOT_ACCEPTED, original scopedL4 and
all prior results unchanged. The L4-T goal remains ACTIVE.

Source read scope: authors' official README and complete BalancedSoftmaxLoss.py,
not full paper/theorem. https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
Paper metadata from that README: Balanced Meta-Softmax for Long-Tailed Visual
Recognition, Ren et al., NeurIPS2020, arXiv2007.10740. No original T claim.
