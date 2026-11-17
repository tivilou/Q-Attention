# Uniform probability mixing does not preserve the native family

This is a limited interpretation result, not a new selected research method. It resolves the uniform-mixture question left open by R50. It preserves the existing scoped L4 result.

For fixed value map, decoder and bias, let F contain all native Q-RMEA probability functions obtained by varying the preparation and effects. The domain here is the admitted variable-cardinality head API: arbitrary valid finite H/R feature tensors and masks, including the specific four-row input below and its eight-row complete duplication. One common parameter setting must define a function on both inputs. These tensors are not certified outputs of the marked frozen encoder and role pipeline.

For any native parameter setting and class r, write

    z_r(X) = b_r + sum_i w_r(H_i,R_i) u_r(H_i) / sum_i w_r(H_i,R_i).

Weights and utilities are pointwise functions. Duplicating EVERY complete valid H/R/mask row k times multiplies both sums by k. Thus the entire 40-logit and probability vectors are unchanged. Entity rows must also be duplicated. This is an exact-real mathematical property, not a bitwise floating-point claim. It also holds for all the stated classical controls, even if their value/decoder parameters vary.

Define a different predictor M by uniformly averaging complete probability vectors after context-only alpha permutations, including identity. All classes use the same index permutation; entity weights and all utilities remain at their positions. This hypothetical ensemble is distinct from the actual one-hash deterministic replay and its CE contrast.

Construct four valid head rows A,B,S,O, with H equal to the first four coordinate vectors in R^1024. A,B are context; S,O have subject/object membership bits. Set the value map's only nonzero coefficient W_value[0,0]=1, its bias zero, and the decoder's only nonzero coefficient decoder[0,0]=1. All class biases are zero. Hence target utilities are (1,0,0,0), and the other 39 utility rows are zero. These are permitted finite parameters of the actual nn.Linear(1024,64) and unrestricted 40x64 decoder, as checked in the pinned source by the primary model; no native forward is claimed.

Set all effects and preparation parameters zero except prepare.W[0,1]=atanh(1/2). The angle transformation pi*tanh makes the first-layer first-wire RY angle pi/2 on B and zero elsewhere. All remaining angles are zero. The common CNOT rings fix |000> and preserve orthogonality of its other basis branch. Therefore all class Born-score rows are

    q = (1, 1/2, 1, 1),  epsilon = 1/1000000.

Let D=7/2+4epsilon and

    a=(1+epsilon)/D=1000001/3500004,
    b=(1/2+epsilon)/D=500001/3500004,
    m=(a+b)/2=750001/3500004.

The native target logit is a on both the input and every complete replication. Other logits are zero. For the probability f(z)=exp(z)/(exp(z)+39), the native target probability is f(a).

At replication k=1, uniform context permutation produces logits a,b with masses 1/2,1/2. At k=2, it chooses two of the four context-weight positions for the two A utility positions. Logits a,m,b have masses 1/6,2/3,1/6. Consequently

    M_1 - M_2 = [f(a)+f(b)-2f(m)]/3 > 0.

The strict sign uses f''(z)=39 exp(z)(39-exp(z))/(exp(z)+39)^3>0 on [b,a] contained in (0,1). It does not assume that softmax is globally convex. Every other class has probability (1-f(z))/39, so the full probability vectors differ. Every candidate member of F has equal probabilities on these two replicated inputs. Thus this particular M is not a member of F on this two-point support. The same obstruction applies to the stated normalized classical head families. Some other mixtures, such as equal-weight or constant-utility cases, can remain native; the result is existential.

For general k, J, the number of high weights assigned to the k A positions, has probability C(k,j)^2/C(2k,k). Then Z_k=b+(a-b)J/k, E Z_k=m and Var Z_k=(a-b)^2/[4(2k-1)]. If c_min and c_max bound f'' on [b,a], Taylor's theorem gives

    c_min Var(Z_k)/2 <= E f(Z_k)-f(m) <= c_max Var(Z_k)/2.

This positive Jensen bias tends to zero relative to f(m). It does NOT tend to zero relative to native f(a), since m<a. The generic Jensen/Taylor/hypergeometric facts are not claimed new. Exact enumeration of 2+24+720 index permutations and rational utility algebra passes; decimal probability values are illustrations, not interval certificates or executed Torch predictions.

An isolated adversarial reviewer found no fatal mathematical defect, with a conditional head-API verdict. Its source-realization qualification is preserved. The primary resolved the explicit value/decoder premise against the pinned AST and sparse exact algebra, clarified the Jensen reference and retained the existential variable-cardinality scope. This does not turn the review into independent source/runtime certification.

The nearby original [Zhang and Xie, IJCAI2020](https://www.ijcai.org/proceedings/2020/0194.pdf), read on pp1-3 with identity/theorem pages rendered, already studies normalized attention's failure to preserve multiplicities. We use only its known duplication collision, independently proved above, not its broader injectivity characterization or unread appendices. Therefore replication invariance itself is not a novelty claim. No exhaustive novelty search was completed for the combined mixture counterexample.

This result prevents importing a risk argument about a probability ensemble as if it guaranteed an optimizer inside the same native head family. It does not invalidate risk inequalities for the ensemble itself, refute task utility of a separately specified ensemble, or explain the actual fresh5 CE contrast. It does not distinguish quantum from real-overlap or softmax; no selected L5 candidate follows. The next theory question should return to the actual class-by-token operator and signed 40-class loss rather than add another diagnostic wrapper.
