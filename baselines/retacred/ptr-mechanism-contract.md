# PTR source mechanism contract — adapter pending

Active source revision: `76f4685f4204f9d469ba1f82ce1dd02ad52aab42` in
[official PTR](https://github.com/thunlp/PTR). Metadata revision is separately
locked to `420926b83025f313f27d77dbb5c74d7089121b89` in `ptr_resources.py`.
This is a source contract, **not an implemented/certified model**.

## Tokenization and vocabulary

`templating.get_temps` constructs three template fragments: `the <mask>`,
three mask tokens, `the <mask>`. Subject and object names are inserted to form
the prompt appended to the original sentence. `data_prompt.REPromptDataset`
uses the five masked positions to collect hidden states. Per-position
verbalizer token sets are sorted; the relation-to-five-verbalizer mapping
must preserve the recovered file, including unusual wording.

Labels have the same canonical names as Improved Baseline but **different
numeric IDs**. Preserve the PTR mapping, align metrics by relation names.
All relation templates must be checked to have identical mask/token structure;
test target changes must not change the model's input.

The released tokenizer sets every `input_flags` entry to zero. `Model.forward`
computes its extra embedding/MLP branch but `torch.where(flags>0,...)` selects
raw token embeddings everywhere. Do not turn that dormant branch into a new
soft prompt and call it the released PTR model. Right-tail truncation is an
upstream fact, not permission to silently change sequence construction.

## Forward and loss

`modeling.Model.forward` uses the source encoder and its word embedding table.
At five masked positions, each hidden state is projected against its position's
verbalizer embedding rows, producing five lists of **raw logits** `z_j`.
`run_prompt.py` computes relation scores:

```text
s[b,r] = sum_j z_j[b, relation_to_verbalizer[r,j]]
L = (sum_j CE(z_j, relation_to_verbalizer[y,j])) / 5 + CE(s,y)
prediction = argmax_r s[b,r]
```

The second, relation-level CE is present in released source. Averaging only
the five prompt objectives would silently change the trained model. Do not
replace sums of raw logits with products/probability sums or output filters.

## Training recipe and certification boundary

`optimizing.get_optimizer` has two disjoint AdamW optimizers: encoder parameter
groups with no-decay exceptions for bias/LayerNorm, and extra embeddings/MLP.
The official Re-TACRED shell specifies RoBERTa-large, sequence512, batch8 per
GPU x8, five epochs, encoder LR3e-5, extra LR1e-5, warmup500, weight-decay.01,
epsilon1e-6. Each has its own linear scheduler; gradient clip is1. Final epoch
is tested, while development scores are tracked. The complete project
protocol and resume states for **both** optimizers/schedulers remain to be
implemented and separately declared, not borrowed silently from Improved.

Required fixture gates: upstream tokenization + input target-invariance,
five masked positions/verbalizer identity, five logits/relation logits/loss,
gradient and both optimizer updates, checkpoint with both optimizer/scheduler
states, and actual train/test intermediate tensor traces. The reversed variant
requires its original entity/relation transformation; this normal metadata
recovery does not define it. No formal queue admission until these pass.
