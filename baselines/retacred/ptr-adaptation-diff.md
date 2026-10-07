# Normal PTR adaptation boundary

Pinned upstream forward, template construction, verbalizer sorting, tokenization,
five-objective plus relation loss, and `get_optimizer` are executed directly.
The objective is compiled from the original AST block; top-level training,
dataset `eval()` and file writes are never executed.

Allowed changes: canonical structured JSON -> source entity-name interface;
one exact `.cuda()` -> selected-device placement; isolated already-frozen args;
local-only pretrained files and eager attention; real train/test numeric capture;
explicit common logical batches and safe post-update checkpoints. Two optimizer
and two scheduler states are mandatory named members (`encoder`, `extra`).

Protocol: five epochs, logical64, default physical4, final epoch selection.
The original script uses 8 GPUs × physical8; our micro accumulation and dropout
partition are a declared difference, not bitwise equivalence to that run.
Retain encoder AdamW3e-5, extra AdamW1e-5, eps1e-6, encoder weight decay.01 with
bias/LayerNorm exclusions, extra decay0, warmup500 and clip1. **FP32** remains
FP32; AMP is not introduced to force an identical protocol across architectures.

Tiny update gates override warmup to0 solely to prove a nonzero parameter update;
separate tests check the source warmup500 and both state restores. Formal train
always uses warmup500. Tiny64-token fixtures are explicitly separate from the
formal512-token contract. Checkpointing is opt-in, frozen per run, tested with
and without recomputation. Changed micro partition requires a new run.

Full tensor evidence includes actual raw/selected embeddings, encoder leaf I/O,
Q/K/V, attention score/probability transformations, five masked hidden states,
tied verbalizer decoder matrices and logits, 40 relation scores, probabilities,
loss and train activation/readout parameter gradients. No extra-token activation,
replacement pooling/classifier, missing relation CE, target leak, output repair
or fabricated reversed transform is allowed. Private tensors never enter Git.

Formal full-weight/GPU gates and measured score reproduction remain pending.
