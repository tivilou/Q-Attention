# MetaEntail general-domain recipe evidence and guard

Status: paper / not_verifiable. This is a recipe intake/audit, not a trainer,
new admitted baseline or recovered RoBERTa author recipe. The existing neural
foundation and model-specific formal blocks are unchanged.

## Recovered evidence versus unresolved scope

The official ACL paper is the already hash-locked2025.naacl-long.165 PDF.
Native PDF text is regenerated, not read from an unverified saved text file.
Section5.3 names RoBERTa-MNLI for general-domain experiments and preserves entity
surfaces. AppendixA.2.4 points to Table6 under a generic MetaEntail heading,
but Table6's caption explicitly scopes its values to BioLinkBERTlarge. These
two observations are both retained: cross-domain transfer is ambiguous, not a
demonstrated prohibition and not a certified general-domain training recipe.

Table6 lists epochs3, batch32, accumulation1, length1024, LR2e-5,
seeds41..45, AdamW, warmup0 and scheduler training steps1000. It does not identify
the batch expansion unit, scheduler kind, AdamW backend/betas/epsilon or general
checkpoint-selection law. Do not silently infer these from framework defaults.
The general-domain checkpoint has512 total pair-token capacity;1024 requires
an architecture change. Fixed1000 scheduler steps differ from E*ceil(ceil(P/B)/G)
for arbitrary expanded feasible pairs and may end LR changes before training ends.
The audit adds no position extension, scheduler substitution or final-run recipe.

The paper's author link is github.com/wphogan/metaentail-re. Its public GitHub
API returned404 at the receipt timestamp. The exact receipt is hash-locked;
deletion versus privacy or transient availability is not inferred. No private
access, fork substitution, repository retry loop or new paper download occurred.
This check does not prove that the authors never released code.

## Explicit proposed recipe boundary

`validate_declared_recipe` requires every field; it never injects defaults or
claims source fidelity. Provenance must be declared_common_protocol with a
nonempty explanation. Callers must supply epochs, pair batch and accumulation,
LR, AdamW backend/betas/epsilon/decay, scheduler kind and budget law, warmup,
pair capacity, overflow handling, clipping choice, seed, selection and AMP.
Only the explicit proposed torch_adamw/linear path is accepted here; it is not
registered with the formal worker or substituted for author code.

The validator reuses the existing expanded_pair_budget instead of maintaining
another version of batch arithmetic. It counts NLI pairs, retains tails, flushes
at epoch end and divides the update loss by its actual pair count. Fixed table
step counts, overlong1024 inputs, test-based selection, implicit default fields,
silent overflow drops and author-recipe provenance claims are rejected.
Accepted proposed input is still paper_interpretive, runtime/length coverage
pending and formal_admission=false. A validation fixture is not a training run.

## Offline entry and remaining work

`bash scripts/check_retacred_metaentail_recipe.sh --describe` reads no model,
Python source cache or PDF. Default execution verifies the existing cached PDF
and locked repository receipt, invokes local pdftotext, and prints scope/gaps.
PYTHON_BIN resolves from the caller environment, then python/python3.

Optional paired arguments --recipe-file and --candidate-counts validate a fully
declared hypothetical protocol; count-file values are feasible training-pair
counts1..40, not raw data or fabricated measurements. This does not recover the
author recipe or launch a job. Exit2 explicitly means unresolved admission;
exit1 is invalid evidence/input. No backend imports, network, upload, Git,
checkpoint writes or formal experiments occur.

Before admission: author general-domain recipe or an explicitly approved
paper-interpretive protocol, independent mechanism review, actual all-example
abstention evaluator, input-length coverage, neural/optimizer/dropout/AMP
evidence, exact resume and complete selected train/test numeric tensors.
Reported94.3 remains unverified and is not a project result.
