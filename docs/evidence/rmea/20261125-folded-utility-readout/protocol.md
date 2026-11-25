# Folded utility readout: fixed small-data single-seed screen

Gen92 Q can slightly beat D/G when all share the same frozen utilities, yet all
absolute scores are poor. Gen91 extra second-order functions helped Q locally
but helped Gaussian more. Neither result establishes co-learning as the cause.
Hypothesis: a redundant factorized affine value/decoder may impede the native
quantum router/readout's joint optimization; remove that redundancy without
changing its real-arithmetic model function class. This is a test, not a cause.

The original source is value(h)=Vh+c, decoder rows D, output bias b, and normalized
valid-word alpha independent of V,c,D,b. Its logit is
sum_i alpha_ri (D_r V h_i) + b_r + D_r c.
Replace with independent M=DV and beta=b+Dc, logit=sum_i alpha_ri M_r h_i+beta_r.
At initialization map exactly from each native seed13 model. In real arithmetic
this is exact for every admitted h/roles/mask, not just a sampled fixture.
Conversely any40x1024 M can be represented with D=[I40,0], V=[M;0], c=0, b=beta.
Thus width64 gives equal function classes at the same router; no rank expansion.
Float32 reduction order differences must be bounded, not called bitwise parity.

This is known linear algebra, not new T. Nov17 R14 readout-class equivalence and
R56 parameterization-sensitive AdamW one-step example already establish relevant
boundaries; do not rediscover or claim novelty for them. Here the untested part
is actual fixed-small-data utility of a direct parameterization, not that proof.
AdamW updates, clipping and decay change under folding. Do not claim the same
regularizer, identical optimizer path, improved conditioning as a theorem or a
purely quantum cause. Fewer registered parameters do not imply equal active
capacity to original controls, smaller saved files or hardware speedup.

Three new fits: Q, native dephased, anisotropic Gaussian base; each uses identical
direct M/beta initialization and identical router mapping to its own native
endpoint. All train jointly from step1; no uniform warmup or frozen utility and
no rescue of the closed256/768 profile. Three new arms53828 trainable scalars:
41000 direct utility plus12828 router. Original value/decoder/bias initialization
tensors retained only as nontraining buffers for unchanged core identity checks;
they are never used to compute the live direct readout. No new data/features,
independent classifier residual, readout nonlinearity or classical/Q mixture.

Reuse exact matched nativeQ/nativeD from gen90, Gaussian base from gen85 and
Gaussian-covariance from gen91. The latter is a practical stronger function-class
reference, not a capacity/optimization-path-matched folded-readout ablation.
Native full control rows are not substituted for small-screen rows.

Fixed original TRAIN8192fit/4096assessment cohort8fb8393b/all40/sentence-disjoint,
seed13/1024updates/b128/micro4/FP32-complex64/TF32off, original AdamW lr.001/wd.01/
betas.9,.999/eps1e-8/clip1. All three fits finish before final1024 assessment.
Primary all40Macro, all40 class/support/confusion/CE/learning history retained.
Qfold must strictly beat the two new arms AND all four reused references: six
fixed gates. Pass admits full single-seed design only; otherwise defer exact
fold/LR/scaling/initialization/seed/budget/control profile, no optional seeds.
Negative small screen does not prove family impossibility.

Stable screen50ce858f and host238f7e68 unchanged, two newly allocated GPUs/ready
queue,3072 physical updates,900sec/arm/3600sec campaign. Only new fold output,
gradient, NumPy Gaussian, padding, actual8 cloned update and zero baseline-buffer
mutation checks; no core/optimizer/resume/COS lifecycle matrix. Short restartable
screen preserves local best/latest and compact Git evidence, zero new COS.
Task weights/features/arrays stay on managed GPU. Historical results and old
scopedL4 unchanged; new L3/L4-T NOT_ACCEPTED, goal ACTIVE.
