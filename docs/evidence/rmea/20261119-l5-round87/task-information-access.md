# Task information depends on the actual readout interface

Status: ANALYTICAL_DRAFT_FOR_EXACT_REVIEW. A separate assessment carries the
completed review of these frozen bytes. This is a transfer boundary, not a
selected method, original quantum-information theorem or L5 contribution.
No model, numerical math fixture, training, scheduler or COS operation ran.

## A. Source-grounded bottleneck example: information and one-use decision differ

Hayashi/Yang, arXiv:2208.10342v2, p10 Lemmas6-7, supplies a fixed-memory example
where a quantum information bottleneck is strictly better than a classical one.
Specialize its uniform noiseless classical source to X=Y in {0,1,2}, memory
dimension2, alpha=1 and beta>1. Entropies below use base2. The objective is

    f=I(T:X)-beta I(T:Y)=(1-beta)I(T:Y).

Its qubit construction is, up to a global phase and basis indexing,

    |psi_j>=(|0>+omega^j |1>)/sqrt(2), omega=exp(2pi i/3).

The average density is I_2/2, every conditional state is pure, and I(T:Y)=1.
This is optimal since cq mutual information is at most log2(dim T)=1. Hence
f_quantum=1-beta. For a classical binary message, mutual information is maximized
by a deterministic channel: for fixed input distribution it is convex in the
channel, and an extreme point is deterministic. A nonconstant binary partition
of three uniform labels has masses1/3,2/3. Thus

    f_classical=(1-beta)h2(1/3),
    f_quantum < f_classical, beta>1.

Now impose a different **operational readout contract**: receive one copy of T,
perform one POVM {M_0,M_1,M_2}, and output one label, with no exact state
description, repeated copies or ensemble-expectation oracle. For any states
rho_j in dimension d and q uniform labels,

    P_correct=(1/q)sum_j Tr(M_j rho_j)
             <=(1/q)sum_j Tr(M_j)=d/q,

because 0<=rho_j<=I and sum_j M_j=I. For the trine ensemble q3,d2, take
M_j=(2/3)|psi_j><psi_j|. These form a POVM and achieve2/3. A classical binary
message also achieves2/3: a deterministic binary partition permits two distinct
correct label guesses among the three uniform labels. Therefore both optima
are exactly2/3 despite the strict bottleneck-objective gap.

This is a **newly checked application of known bounds** to the cited source
example, not a novelty claim or a refutation of that paper. Its p11 Eq60-62
classifier evaluates state features/Hilbert-Schmidt kernels; p12 reports a
different noisy synthetic classification experiment. Such access is not the
single-copy POVM contract above. An exact classical representation of density
matrices is allowed for this project, but its access and computational cost
must be stated. Neither the source objective nor our dimension bound predicts
Q-RMEA Macro-F1, rejects all bottlenecks, or establishes a quantum speedup.

Datta/Hirche/Winter, arXiv:1810.03644v3, p2 Eq4 and pp3-4 operational setting,
likewise specifies channels, shared entanglement, asymptotic iid source coding,
decoder side information and fidelity. That source-coding operational theorem
is not a classification-risk theorem or a finite-memory fixed-circuit guarantee.
No rate inequality or full convergence/source-coding proof is transplanted here.

## B. What the implemented model actually observes

For this section n is the number of valid tokens, and every token sum/register
below includes only those tokens. The quantum branch has M_r=|phi_r><phi_r|;
the dephased branch instead has M_r=diag(|phi_r,k|^2). The supplied RMEAHead
computes p_rt=Tr(M_r rho_t), adds EPS, normalizes
over valid tokens, computes classical signed utilities u_rt from value+decoder,
and outputs sum_t alpha_rt u_rt+bias_r. Neither a measurement instrument nor a
post-measurement state appears in that executed source.

Consequently, holding p, u, mask and bias fixed while changing only a hypothetical
instrument leaves every old logit fixed. This is source correspondence, not a
new theory of instrument independence or an empirical intervention result.

For precision about possible future readouts, let a trace-nonincreasing CP map
I_E(X)=sum_a K_a X K_a^dagger have effect E=sum_a K_a^dagger K_a. If E=e|phi><phi|
has rank1 and e>0, positivity forces K_a v=0 for every v orthogonal to phi:
sum_a ||K_a v||^2=<v|E|v>=0. Thus K_a=|v_a><phi| and

    I_E(rho)=<phi|rho|phi> sum_a |v_a><v_a|.

Whenever its success probability is positive, the normalized **output of this
system alone** is the same fixed state sigma=sum_a |v_a><v_a|/e, independent of
rho. This standard rank1-instrument fact does not imply that correlated side
registers lose all information.

For any joint state rho_RQ and an instrument acting only on Q,

    Tr_Q'[(id_R tensor I_E)(rho_RQ)]
      =Tr_Q[(I_R tensor E)rho_RQ].

This follows by partial-trace cyclicity or testing both sides against arbitrary
observables on R. The reduced conditional state of R depends on the effect E,
not its choice of instrument. R may still retain input-dependent information.

For example, take a **classical** token register,
rho_RQ=(1/n)sum_t |t><t| tensor rho_t. The exact smoothed native attention can be
represented using the valid effect

    E_EPS=(M_r+EPS I)/(1+EPS).

In the quantum branch its spectrum is1 along phi_r and EPS/(1+EPS) in the
orthogonal complement. In the dephased branch the eigenvalues are instead
(|phi_r,k|^2+EPS)/(1+EPS). The normalization identity holds in both branches.
The reduced conditional R distribution is exactly
(p_rt+EPS)/sum_s(p_rs+EPS). Measuring its diagonal utility observable yields the
original logit minus bias. This is a separable cq representation and a known
conditional-probability identity, not an entanglement or steering witness.

Important exception: EPS>0 makes E_EPS full rank. The rank1 output-state lemma
does **not** apply to a physical instrument implementing that full-rank effect.
The native code adds EPS classically and supplies no physical instrument; this
equivalent effect representation confers no executed hardware implementation.
For either effect, changing an unobserved instrument cannot improve the old
function. Making its output operational, adding correlated registers or choosing
an explicit phase-sensitive readout changes the architecture/access contract
and requires its own utility evidence. Coherent token registers retain the R85/
R86 phase-preparation boundary; they are not silently inferred from this cq state.

## Consequence for the current research route

All correspondence claims above concern exact real/complex arithmetic, not
bitwise floating-point identity or executed physical instruments. The empirical
history and R85/R86 results below are external project context; they are not
new findings independently verified by this mathematical/source review.

The positive empirical carryover remains ordinary trained-head utility and local
weight-to-signed-value alignment, with the frozen scopedL4 limitations. The
current information-retention proposals have no demonstrated missing task
structure and no selected new causal mechanism. Generic mutual-information
increase, different instrument backaction, or more states in a Bloch diagram is
insufficient evidence of improved relation decisions.

A future proposal must state its accessible variables/registers, decoder and
cost; identify task structure lost at an **observed** link; and predict a contrast
against a matched alternative at that link. Existing independent ideation NONE
and all stopped families remain. No bottleneck loss, rank retune, new training,
phase grid or formal handoff is selected. L5 NOT_ACCEPTED.
