# Finite-circuit counterexample and the remaining scientific question

This is a synthetic counterexample using known coadaptation/representation
equivalence, not an L5 contribution or an explanation of the observed NLP result.
It strengthens the earlier mixed-effect caveat by staying inside the original
pure-effect circuit and the actual affine1028->12 preparation parameterization.

Take two head-boundary inputs with identical token feature bags
[e0,e1,0,0] in R1024, all four positions valid. Positions2/3 are subject/object
anchors in example0 and object/subject in example1. Construct the four role
features by the original clipped relative-distance /32 and membership recipe.
Their first-channel minus second-channel distance is +1/32 on all four tokens
for example0 and -1/32 for example1; clipping is inactive.

Let k=atanh(1/2). All preparation preactivations are zero except:
first-layer wire0 RY bias k;
first-layer wire0 RZ preactivation k*(h1-h0);
second-layer wire1 RZ preactivation 32k*(role0-role1).
Every actual angle is pi*tanh(preactivation), strictly inside(-pi,pi).
The first CNOT ring maps |100> to |011>; the second maps |011> to |010>.
The two RZ phases add on the active branch. Therefore the two context states
are psi_plus=(|000>+|010>)/sqrt2 and psi_minus=(|000>-|010>)/sqrt2,
with their assignment exchanged in example1. The entity anchors have relative
phase +i or -i and overlap probability1/2 with either real effect.

HeadA uses the shared pure effect phi_plus. It is realized by first-layer
wire0 RY=pi/2 and every other angle0. HeadB uses phi_minus: add first-layer
wire0 RZ=pi/2 and second-layer wire1 RZ=pi/2. All are finite original circuit
parameters. Effects are global per relation; no joint-POVM constraint applies.
This explicit two-RZ construction fills a gap in the independent advisory's
earlier shorthand, which did not specify an affine map from valid role inputs
to all desired preparation states. The original advisory is retained unchanged.

For eps=1e-6 and Z=2+4eps, the first example's allocations are
[(1+eps)/Z,eps/Z,1/4,1/4] for A, and its first two entries exchanged for B.
The second example exchanges them again. Set the first value coordinate to
Z*(h0-h1), the other63 to0. HeadA's active decoders are+1/-1; HeadB's are
-1/+1. Other38 decoders are0 with biases-100. Own logits are(1,-1) for
example0 and(-1,1) for example1 in both heads. Thus their pooled features are
related by a fixed coordinate permutation and their task information is equal
on this two-point input distribution, for any mixture probability.

Either attention transplant into the recipient's fixed utilities exchanges its
active logits, making both predictions wrong. Reversing the recipient readout
recovers the original logits. Log-sum-exp is unchanged by the exchange, so the
cross-entropy damage is exactly2 per example even with38 dummy classes.
Uniform pooling of these identical bags cannot distinguish these two inputs.

This is a head-input construction, not an output of RoBERTa or Re-TACRED.
The actual encoder inserts ordered subject/object punctuation before word
features are built; therefore uniform's lack of direct role input does NOT
establish that the real uniform baseline lacks entity information. No such
empirical attribution or full-function equivalence is claimed.

For the actual affine value map Vh+c, decoder d_r and normalized allocation,
z_ir = w_r^T mu_ir + beta_r, mu_ir=sum_t alpha_irt h_it,
w_r=V^T d_r, beta_r=b_r+d_r^T c. Since the value width64>=40relations, every
40x1024 effective linear readout W is representable: put W in the first40 rows
of V and use the first40 coordinate vectors as decoders. This familiar matrix
factorization observation does not equate training implicit bias, penalties,
conditioning or optimizer budgets.

The remaining discriminating question is whether a difference survives a fair,
train-only readout fit evaluated on separate inputs. Same-sample interpolation
is a confound:1024 examples with1024-dimensional pooled features plus bias may
fit arbitrary in-sample logits. A residual gap would still require separating
conditioning/optimization from information loss; it is not automatic Born
necessity. Existing exposed validation cases are not independent confirmation.

Known-neighbor boundaries: free attention/output-nullspace analyses and
representation-equivalence/probing already cover the general idea. The ICLR
2025 normalized-attention identifiability result quantifies over all inputs
of a different model; a two-point counterexample does not contradict it.
Compressed-sensing tomography assumes a specified random operator ensemble;
task/value-dependent operators are not automatically covered.

No theorem novelty or new task performance is accepted. The owner run is a
two-example source-faithfulness check through the shared CPU scheduler only;
there are no dataset/checkpoint/cache/COS reads or trained updates. All inputs
and weights are predetermined finite synthetic values, with no parameter search.
