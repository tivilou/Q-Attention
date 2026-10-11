# Imported Improved Baseline mechanism contract

The four implemented marker configurations share this contract; their exact
`input_format` and per-model evidence limits are recorded in
`manifests/<model>/baseline-manifest.json`.

## Source facts

Official source: `wzhouad/RE_improved_baseline`, revision
`fe00781116ab4c34cbcd9a2ae5b02513afae87ee` (MIT).

1. `prepro.RETACREDProcessor.tokenize`: encode the sentence with the selected
   entity marker scheme, retain subject/object marker indices and the upstream
   40-label mapping; `utils.collate_fn` supplies padding and masks.
2. `model.REModel.forward`: encoder output `H[B,L,D]`, select `H[b,ss[b]]` and
   `H[b,os[b]]`, concatenate to `[B,2D]`. This is **not CLS pooling**.
3. Classifier: Linear(2D,D), ReLU, Dropout(.1), Linear(D,40). CrossEntropyLoss
   consumes `logits.float()` and labels. Argmax alone selects a relation.
4. Tokenizer typed markers are added during encoding. Resize the pretrained
   embedding table after encoding the declared splits, before optimizer setup.
5. Upstream CUDA autocast and GradScaler are retained. Eager attention matches
   the pre-SDPA upstream path and exposes real attention softmax I/O.

## Separately declared common protocol

`train_retacred.py` is provenance evidence, **not the executed training loop**.
The project loop is `baselines.retacred.train` with five epochs, final-epoch
selection, default logical batch64 / micro4, weighted accumulation, AdamW
LR3e-5 / eps1e-6 / weight-decay0 / correct-bias, clip1 and 10% linear warmup.
Unlike upstream, it consumes the remainder, updates after a complete logical
batch, and never uses periodic test evaluation to select a checkpoint.
Physical batch and checkpointing are frozen because dropout partition changes
are not bit-exact. There is no EMA, noise schedule or generative sampler in this
supervised relation classifier; those gates are explicitly not applicable.

The comparison is a common-protocol adaptation, not a reproduction of a
reported score. This budget is also different from the frozen-encoder core
Q-Attention baseline; do not infer mechanism-specific benefit from that
cross-budget contrast.

## Evidence and certification limit

`tests/test_retacred_reference_suite.py` uses tiny RoBERTa plus the official
tokenizer and directly imported source. It checks forward, loss, gradients,
AdamW updates, training-mode dropout tracing, numeric intermediates and exact
pause/resume. These are mechanism/behavior fixtures, **not a real
RoBERTa-large CUDA/AMP or published-result reproduction**. Checked-in
manifests therefore remain `not_verifiable` for full-weight formal
certification until the real-weight, CUDA/AMP and protocol gates are recorded.
`fidelity.py` emits a separate bounded runtime comparison record; its scope
must never be broadened into a full formal certificate or a measured score.
