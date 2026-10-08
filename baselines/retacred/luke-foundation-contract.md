# LUKE foundation mechanism contract

## Status and boundary

Prepared adapter, not_verifiable. Not admitted as a new selector, not a replacement
for LUKE+CoRE/CFIE/Resample/Focal, not a reproduction of reported90.3/90.9.
Four LUKE cohort rows stay blocked and default five configurations are unchanged.
Task data remain the existing qualified canonical40-class Re-TACRED splits.

## Locked source facts

Official studio-ousia/luke revision251654548a15b9a8ddc708a53b715bdf3a112102.
Fifteen selected source files have SHA256 and original Git blob identities in
luke_foundation.lock.json. The small public source cache is private .external.
Author reader parse_tacred_file inserts subject <ent> and object <ent2> markers
in sentence-position order, preserving semantic subject/object roles. It mutates
decoded token lists; the adapter binds in-memory JSON to a private deep copy and
executes the exact hash-locked function, not a rewritten sentence formatter.
Labels and sample ids do not affect the encoded model input.

RelationClassificationReader.text_to_instance records first/last marker token
positions; subject/object entity ids come from [MASK]/[MASK2] in entity_vocab.
EntityBasedRCFeatureExtractor uses both inclusive spans, including markers,
and pads position lists to the batch's maximum with -1. A blanket HF
task=entity_pair_classification call can impose max_mention_length; this adapter
uses task=None and explicitly routes source-equivalent inclusive positions.
The actual AllenNLP BPE/indexer equivalence gate remains pending, not passed by
the deterministic tokenizer fixture.

Entity-aware attention=true. Author embedder returns entity_last_hidden_state.
The two entity representations are concatenated in subject then object order,
dropout0.1, biased Linear(2*hidden_size,40), and single-label cross entropy.
The actual author HF conversion classifier is imported from hash-locked bytes;
its active forward is not replaced. Source AlleNLP classifier initializes with
nn.Linear's default uniform law; adapter resets newly initialized classifier to
that law, rather than retaining HF post_init normal initialization. RNG call
order/common label-index order differs and is a declared adaptation.

## Checkpoint identity (metadata, not a local load certificate)

studio-ousia/luke-large at0729d044dfe301d9ecabc222d60633f92ac450eb,
not studio-ousia/luke-large-finetuned-tacred. Pretrained architecture LukeForMaskedLM:
word50267, entity500000, entity width256, hidden1024,24layers/16heads.
Provider's pytorch_model.bin size2246566835 and SHA256
9ea73dbb7e0625e88c646f26230902c3df000e71320139e99b0703e1654f7bcc.
The historical foundation lock records metadata-only preparation. Subsequent
asset evidence at16b122e verifies eight originals and twelve mirrored parts;
that resource verification does not constitute a model load/runtime certificate.
Eight required assets are identity-locked. Constructor streams local size/hash
validation, prohibits remote fallback, validates loading-key sets, preserves
pretrained entity attention, and creates the only new40-class head.
Allowed missing keys: new classifier weight/bias and unused LUKE pooler;
allowed discarded pretraining keys: lm_head.* and entity_predictions.*.
Other missing/unexpected/mismatched/loading errors stop initialization.

## Training and evaluation remain separate

Official base AllenNLP recipe:5epochs, physical4/accumulation8, AdamW1e-5,
weight decay.01 except bias/LayerNorm.weight, warmup.06, patience3,
dev +micro_fscore, seed0. README optionally enables CUDA AMP. These are not
the recovered CoRE authors' Re-TACRED recipe. Common logical batching,
canonical evaluator, seed/budget/checkpoint-selection and optimizer/scheduler
binding must be declared and verified before runner admission. No source
challenge-set filtering, label-only rule or CoRE coefficient is added here.

## Evidence and pending gates

Tests execute actual source parser and actual author classifier forward with
tiny synthetic vectors/scalar CE/dropout identity. These prove interface/head
algebra only; they do not prove backbone, gradients, weights, dropout or optimizer.
No Torch/NumPy imports or full formal run occur in preparation checks.
Full-weight constructor checks actual cgroup/host headroom before Torch import
or large weight reads, requiring12GiB for bounded initialization. This is a
declared conservative preparation gate, not a measured GPU performance claim.
It never adjusts container limits, kills a process or clears another process.
The current1GiB cgroup blocks full-weight checks cleanly before allocation.

Before admission: real BPE/indexer, entity-attention fixed-input forward/loss,
gradients/update/scheduler/dropout, actual local checkpoint-key/weight comparison,
exact post-update resume, actual numeric train/test SampleCapture integration,
common-protocol and CUDA/AMP gates. Flags remain false until literal runtime
evidence exists. SampleCapture now accepts both Sequential and Linear classifier
heads and observes LUKE word/entity embedding and encoder container outputs.
The foundation remains outside the formal runner; each runtime evidence file
must say whether actual full weights or only a tiny random architecture ran.

## Bounded runtime validation entry

`bash scripts/check_retacred_luke_runtime.sh --describe` performs no backend,
weight, network or output-file access. `--mode tiny --output-dir PATH` uses the
actual pinned author classifier and actual installed HF entity-aware backbone
with a two-layer random32-wide synthetic-ID architecture. It does not establish
tokenizer/real-weight/AllenNLP/published-score fidelity.

`--mode full --device cuda --output-dir PATH` uses the existing hash-locked
LUKE-large assets and real BPE on exactly two synthetic raw records. It requires
24GiB available host/cgroup headroom before Torch imports or weight reads.
Insufficient memory produces `pending_memory_gate` evidence and exit2. It does
not fetch assets, train on task splits, tune coefficients, publish or push.

Both modes compare direct-source and traced adapter forward/loss, all gradient
hashes and dropout RNG exactly; require nonzero word/entity/entity-aware-query/
head gradients; perform two diagnostic FP32 AdamW updates; and compare model,
optimizer, scheduler, RNG and output after restoring the post-first-update
checkpoint. Actual train/test numeric tensors are saved and reopened. Only one
model is resident: continuation is restored sequentially, not in a second model.

The diagnostic AdamW1e-5/eps1e-8/decay.01, zero-warmup four-step schedule is
explicitly NOT the source AllenNLP training recipe. The nonzero-LR gate avoids
a vacuous first-update test; source optimizer/warmup/common-loop and complete
method/recipe admission remain pending even after diagnostic success.
