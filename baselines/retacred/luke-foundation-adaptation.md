# LUKE foundation adaptation boundary

Allowed: in-memory JSON binding and private copy; canonical40 label order/head
width; local-only pinned asset initialization; requested device; framework-neutral
word/entity batching matching inclusive positions; actual observability and
eventual common-loop resume integration. New classifier uses original nn.Linear
initialization law but RNG call order differs. These are declared, not hidden.

Forbidden: CLS substitution, role reordering by text position, averaged entity
pooling, clipping long mentions to30, silently truncating sequences/entities,
marker repair, entity-vocabulary/word embedding resizing, gold-fed model input,
loading41-label fine-tuned TACRED head, switching off entity-aware attention,
discarding unexpected load errors, using test to tune selection, admitting this
foundation as CoRE/CFIE/Focal/Resample or inheriting their paper scores.

Current fail-fast constraints: exactly40 unique labels including no_relation;
two valid nonoverlapping inclusive spans; no raw marker-token collisions;
BOS/EOS identity; token length<=512; both role markers present exactly twice;
no entity mention length cap. Long/overlapping records require a separately
reviewed adaptation instead of silently deleting samples from evaluation.

Tiny deterministic tokenizer/vector fixtures are interface checks only. Real
AllenNLP-tokenizer and full-backbone/gradient/optimizer/weight/resume/trace/AMP
certification is pending. No trained output repair or postprocessing exists.
