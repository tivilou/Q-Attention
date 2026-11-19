# Finite source-head realization of the cross-fold witnesses

Status: primary analytic construction; independent review is recorded separately.
Same mechanism family, no scientific execution or training contract change.

## Claim scope
This is an expressivity statement about the existing RMEAHead at one shared
finite parameter point over legal 1024-dimensional feature inputs. It does
not prove that a fixed pretrained encoder emits these features, that fitting
finds these parameters, or that actual NLP errors follow the stipulated law.
No existing checkpoint, forward branch, tensor shape or source file is changed.

## 1. Actual circuit and finite parameters
The code uses three qubits, wire0 MSB, two layers of RY then RZ per wire, followed
in each layer by CNOT0->1,1->2,2->0. One ordered ring sends bit triples to
M(a,b,c)=(b xor c,a xor b,a xor b xor c). Thus M(111)=001 and M(M(010))=001.
Put lambda=atanh(1/2)=log(3)/2; pi*tanh(lambda)=pi/2 with finite lambda.

(a) All six RY angles zero, with all RZ zero, yield e0=|000>.
(b) All six RY angles pi/2, with all RZ zero, yield e1=|001>.
After the first rotation layer the state is |+++>. A CNOT ring leaves its
equal positive superposition invariant. RY(pi/2)|+>=|1>, so the second rotation
layer gives |111>, and the final ring gives |001>. No angle pi or limit occurs.
(c) Only the first-layer wire1 RY angle pi/2, all other angles zero, yields
phi=(e0+e1)/sqrt(2): the two rings map 010->111->001 with no relative phase.

## 2. One shared prepare map and all forty query states
Zero every prepare bias. For each of the six RY output rows, set only its
weight on feature coordinate h[0] to lambda. All other feature/role weights
and all RZ rows are zero. Hence h[0]=0 produces e0 and h[0]=1 produces e1.
For each class c, set effect[c,layer0,wire1,RY]=lambda and all its other
stored entries zero. Source pi*tanh(effect) yields the common phi from1(c).
Coincident query rows are a permitted parameter point, not a source change.

## 3. Simultaneous utilities with no conflict in state preparation
Set value.bias=0. The first40 rows of value.weight select h[1],...,h[40];
its remaining24 rows are zero. Set decoder=[I40|0] and class bias=0.
The actual utility einsum is utility_c(h)=h[c+1] for every class simultaneously.
State preparation only reads h[0], whereas utilities read coordinates1..40.

For arbitrary b_A,b_B in R^40 use four tokens (all omitted coordinates zero):
token0: h[0]=0, h[1:41]= b_A;
token1: h[0]=0, h[1:41]= b_B;
token2: h[0]=1, h[1:41]=-b_A;
token3: h[0]=1, h[1:41]=-b_B.
Mask all four tokens valid. Any legal role inputs work because their weights
are zero. States are(e0,e0,e1,e1), alternating folds A={0,2},B={1,3}.
Every query-state squared overlap is1/2; EPS=1e-6 therefore leaves native token
weights1/4 and fold masses1/2. Fold utilities are exactly(b_A,-b_A),(b_B,-b_B).
The prior measured_attention/fold_attention equations thus arise from the
actual shared prepare/value/effect/decoder parameters, not arbitrary unrelated
intermediate tensors. All finite positive/negative amplitude choices are allowed.

## 4. The two prior laws use the same fixed head
For y=7, positive witness: b_A[y]=b_B[y]=s; every distractor has
b_A[c]=xi*t,b_B[c]=zeta*t with independent uniform signs xi,zeta shared across
distractors. Only input features change with the signs; states and parameters
remain fixed. The previous finite sufficient CE window and identical cross/full
argmax ceiling therefore hold at this source-head parameter point over that law.
Negative useful-signal witness: distractors zero, and true-class b_A[y],b_B[y]
independently uniform in{0,2s}. The SAME head realizes the previous reversal.
Changing a class-conditional data law is not a label-dependent inference branch.

Any y could instead carry its coherent signal in coordinate y+1 under these
same parameters. No training label is added to forward; the statement merely
defines a synthetic class-conditional distribution of legal feature inputs.

## Boundaries
Proved here only in exact arithmetic: finite HEAD expressivity for the specified
feature laws. Encoder-manifold realizability, convergence, numerical execution,
NLP noise independence, held-out Macro/CE benefit, literature novelty, quantum
necessity and L5 remain unestablished. Equal cross/full predictions in the prior
symmetric witness still prevent a Macro guarantee. No new algorithm or numerical
qualification is claimed. Frozen source815 training and its single-attempt
scheduler admission retain priority; this proof changes neither gate nor task.
