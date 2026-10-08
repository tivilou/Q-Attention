# MetaEntail neural adapter preparation

Status: paper provenance; not_verifiable; formal admission closed. Checkpoint
bytes/private delivery and a dedicated neural foundation are prepared. Actual
model loading/forward/autograd, the paper's94.3 result and CUDA execution remain
uncertified. The foundation is not dispatched through the formal suite.

## Checkpoint facts

Section5.3 and footnote6 identify FacebookAI/roberta-large-mnli. The official
snapshot is2a8f12d27941090092df78e4ba6f0928eb5eac98, MIT,24-layer1024-wide
RoBERTa. Six small files match both provider Git blobs and locally computed
SHA256/length. The selected model.safetensors is now streamed and hash-verified
in full; seven files are mirrored as nine <=512MiB shards on the authenticated
exchange. metaentail_assets.lock.json binds exact originals/shards; mnli_assets.py
and fetch_retacred_mnli_assets.sh fetch and assemble atomically, preserve corrupt
caches for diagnosis and verify complete original hashes. Six actual head/word/
position tensor shapes and raw byte hashes are inspected without importing a
neural backend; this is not loaded-tensor equality or numerical inference.
Prefer the locked model.safetensors; do not switch to the
project's plain RoBERTa asset or use an unpinned main revision.

Config NLI IDs are0=CONTRADICTION,1=NEUTRAL,2=ENTAILMENT. Training targets must
use this actual label map; relation IDs0..39 never become head indices. Preserve
the pretrained dense/out_proj three-class MNLI head. Planned loader:
RobertaForSequenceClassification.from_pretrained(local_pinned_directory,
local_files_only=True,output_loading_info=True). Do not override num_labels,
resize embeddings, reinitialize the classifier or ignore mismatched sizes.
Missing/unexpected/mismatched keys and exact loaded head/embedding tensor
equality remain required runtime gates, not inferred from config.

## Actual input boundary

Tokenize premise and hypothesis as two strings, preserving raw entity surfaces.
The official fast BPE tokenizer adds `<s> premise </s></s> hypothesis </s>`:
four special tokens. No added entity markers, prefix-space override, entity
abstraction, lowercasing, manual separator assembly or token_type_ids.

max_position_embeddings=514 and pad_token_id=1 imply512 usable non-padding
positions; tokenizer model_max_length is512. Thus a512-token pair leaves508
content tokens across BOTH strings. The paper's Table6 caption is BioLinkBERT;
its1024 setting is not a recovered general-domain RoBERTa recipe. No position
extension is introduced. The encoding preflight keeps full inputs and rejects
an overlong pair with actual/allowed lengths. This rejection policy is an
explicit preparation choice; an eventual full-run overflow policy requires
length coverage evidence and declared adaptation, rather than silently dropping
examples or truncating entity evidence. Shorter limits require explicit input.

Record each selected pair's original sample ID, candidate relation/index,
premise/hypothesis, BPE tokens/IDs, offsets, sequence membership and attention
mask. Targets are created separately by the train-only paper primitives; the
pair encoder reads no gold relation. Inference groups remain gold-independent.

## Expanded-pair budget, not relation-example batches

Let c_i be the number of feasible hypotheses for original training example i.
The train-only type filter must include its gold, so1<=c_i<=40. One expansion
epoch contains P=sum(c_i) NLI training pairs, not N original relations. Flatten
canonical candidate order with a stable `(sample_id,candidate_relation)` identity;
then use an epoch/seed-owned shuffled pair order, recorded in checkpoints.

The proposed boundary uses B PAIRS per mini-batch, accumulation G, no dropped
tail and epoch-end flushing: M=ceil(P/B), U=ceil(M/G), E epochs give E*P pair
presentations and E*U updates. The planner reports final mini-batch/window sizes
and a count-projection hash; it does not stand in for full data/order hashes.
Accumulated loss is sum of pair CE divided by actual pairs in that update
window, including the partial tail. Averaging mini-batch means without their
pair counts would overweight a tail. Equal-original-group weighting would
change this objective and is not silently inserted.

Batch32 pairs is different from32 relation examples expanded to many pairs.
Changing GPU count or physical partition must preserve this declared update
objective and exact dropout/resume contract. Scheduler steps use U*E, not N/B.
Epochs, effective pair batch, learning rate, AdamW details, warmup, selection
and seeds remain unresolved for the paper's general-domain experiment. No
BioLinkBERT/pretraining settings or generic Improved trainer fallthrough are
presented as author recipe. The helper requires explicit epochs/B/G and labels
its output a proposed adapter choice. It does not choose a formal configuration.

## Integration gates still pending

metaentail_neural.MetaEntailFoundation now implements the dedicated local
three-class pretrained loader, group-pair expansion, dynamic pad1/attention0,
native NLI CE forward, group decision, actual-event SampleCapture interface
and classifier parameter-gradient capture. Loader preserves MNLI dense/out_proj;
it checks loading reports and exact source head/embedding tensor equality before
device transfer. Eager attention is explicit for inspectable actual attention
scores/probabilities; no weights or objective change is implied. Numerical
fidelity/trace perturbation evidence remains required. A conservative unchanged
12GiB pre-import gate prevents model allocation in the current1GiB container;
do not read a header/loader implementation as a successful neural runtime test.
The suite still rejects MetaEntail before launching any formal worker.

- Distinct MNLI assets/mirror/original hash checks are prepared. Before formal
  suite admission, bind this route into the owning runner and retain hashes
  before allocation; no weights/tokens in Git.
- A dedicated neural forward on three-way logits plus pair CE, original-group
  decoding through existing grouped_prediction, and all-example metrics with
  abstentions/coverage. Do not use a zero-shot pipeline with alternate scoring.
- Explicit optimizer/scheduler recipe and binding of pair budget; loss, gradient,
  update, checkpoint-head integrity and seed-fixed repeatability evidence.
- Checkpoints per(model,seed) with data/filter/checkpoint/encoding/recipe hashes,
  epoch, post-update expanded-pair cursor, pair-order identity, RNG, optimizer,
  scheduler and adaptive memory state. Reject incompatible resumes.
- Selected train AND test cases must capture actual complete embeddings,
  per-layer states/attention, CLS features, dense/head inputs/outputs, NLI
  probabilities and relation selection; train adds loss/gradient/update evidence.
  Bind every pair to its original group; real numeric tensors go to the private
  exchange with hashes, not placeholders or posthoc reconstructed training.
- Independent paper-mechanism review, full-weight CUDA canary and paper/common
  evaluator reconciliation. The entail-only published denominator and94.3
  remain unverified. Formal workers stay blocked until owning gates pass.

Offline metadata command: `python -m baselines.retacred.metaentail_checkpoint
--metadata-dir LOCAL_PINNED_SMALL_FILES`. It verifies metadata only, with zero
network/model jobs; a passing result does not imply neural runtime fidelity.

Asset command: `bash scripts/fetch_retacred_mnli_assets.sh`; requires the existing
PROJECT_EXCHANGE_URL/TOKEN only for missing files, supports explicit --cache-dir.
Inspection: `python -m baselines.retacred.metaentail_neural --checkpoint
LOCAL_VERIFIED_MNLI --inspect-only` verifies full asset bytes and selected tensor
headers/hashes, without loading Torch. Without --inspect-only the entry attempts
the foundation loader; insufficient verified headroom prints pending_memory and
exits3 before weight reads/backend import. Neither entry performs training/push.

The bounded runtime entry is `bash scripts/check_retacred_metaentail_runtime.sh`.
It records a complete private `.pt` tensor artifact and human-readable JSON for
one selected train group and one selected test group. Each selected group keeps
all feasible candidate pairs, original raw record, premise/hypothesis, BPE IDs,
attention mask, hidden states, attention probabilities, CLS readout, dense and
out-projection inputs/outputs, NLI probabilities and grouped relation decision.
The train artifact additionally contains targets, loss, activation/parameter
gradients, the post-update checkpoint and exact resumed second update. Test reads
no gold relation for the model path and contains no target/loss tensors. The
script uses `--mode tiny` only for a synthetic architecture fixture and
`--mode full` for the pinned model; current project server remains blocked by its
1GiB cgroup before backend import. These are diagnostic artifacts, not formal
results or a substitute for the collaborator's full run.
