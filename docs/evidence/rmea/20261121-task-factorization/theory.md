# Task-output matching with shared nonnegative routing factors

Status: provisional mathematical diagnostic, L5 NOT_ACCEPTED. No new model or experiment. Preserve native Q-RMEA L4 and all historical results. Architecture: native RMEAHead at main f58646df.

1. Motivation and established background

Native p_rt=|phi_r^H psi_t|^2, alpha_rt=(p_rt+eps)/sum_s(p_rs+eps), logit_r=b_r+sum_t alpha_rt u_rt; eps=1e-6. Relations are separate binary effects, not a joint POVM. Values/decoder are fixed in the following analysis.

Dephased raw scores factor as S=AB with A_rk=|phi_rk|^2+eps and B_kt=|psi_tk|^2, k=1,...,8. This includes smoothing because sum_k B_kt=1. Arbitrary B>=0 is a relaxation of normalized circuit populations. A is shared across passages.

PSD/nonnegative score factorizations and rank separations are established: Fawzi et al., arXiv:1407.4095v1, pp3-6, definitions2.2/2.4, propositions2.5/2.8, field distinction2.2. No new rank theorem or quantum speedup claim. A score difference must survive normalization and readout to affect the task.

2. Conditional exact matching theorem

Fix utilities U in R^(R x T), target pre-bias logits z, and A in R^(R x K) with A_rk>0. For nonzero B>=0 define D_r=sum_kt A_rk B_kt and f_r(B)=sum_kt A_rk B_kt U_rt / D_r.

There exists B with f(B)=z iff 0 lies in the convex hull of the K*T columns of M, where M_[r,(k,t)]=A_rk(U_rt-z_r). Equivalently, x>=0, sum x=1, Mx=0.

Proof: homogeneity permits x=vec(B)/sum(B). Positive A makes each D_r positive. Subtract z_r D_r from each numerator: the result is exactly (Mx)_r. Conversely a simplex solution defines valid B. For multiple passages a fixed common A gives separable feasibility problems in passage-local B. Optimizing over A is not covered.

By standard strict convex separation, exactly one holds: a primal simplex solution; or y with delta=min_(k,t) y^T M_[:,(k,t)]>0. This specializes known convex feasibility/Farkas reasoning, not a new theorem of alternatives.

A dual witness implies for all B:
  ||f(B)-z||_infinity >= delta / sum_r |y_r| max_k A_rk.
Indeed delta <= y^T Mx=sum_r y_r D_r(f_r-z_r), while D_r<=max_k A_rk. The denominator is positive for a valid strict witness. Numeric solver status alone proves neither branch: directly verify primal residuals or every dual column product against a declared roundoff bound.

If a simplex x has ||Mx||_infinity<=rho, output error is <=rho/min_rk A_rk. More tightly, inspect each actual residual divided by D_r. With unchanged bias and output error<=eta, a unique native argmax with top-two margin>2eta is preserved; CE changes by at most2eta for any gold label. The latter follows from logsumexp being1-Lipschitz in infinity norm and the target logit changing by at mosteta. Small eps can make the coarse residual bound weak.

A fixed-A dual does not exclude other A or all rank-eight families. Nonzero logit error need not hurt CE or Macro-F1. A primal is an oracle existence result, not a learned generalizing model.

3. Reachable rank-nine counterexample with rank-one task-equivalent routing

Use the actual depth2 3-qubit RY/RZ plus real CNOT-ring topology. Set first-layer angles0. For the second layer choose each RY from{0,pi/2}, RZ0. Before the common final ring these are the eight tensor products of |0> and |+>. Their projectors are linearly independent since {rho_0,rho_+} is independent and tensor products preserve independence. They are real.

Add a ninth state: second-layer RY(pi/2),RZ(pi/2) on wire0, all others0. Its projector has a nonzero imaginary part, independent from the first eight. The common real final ring preserves independence. All angles lie strictly within(-pi,pi), realizable by finite pre-tanh parameters.

Choose nine query states equal to these token states. P_rs=Tr(rho_r rho_s) is their positive-definite Hilbert-Schmidt Gram matrix, rank9. P+eps*11^T remains positive definite, and row normalization preserves rank. Extend to40 relations by duplicating queries. Attention rank stays9; no eight-component nonnegative factorization reproduces this attention matrix.

For every relation choose a nonzero real v_r perpendicular to both1 and d_r=alpha_r-uniform. Such vectors exist: the orthogonal complement in R9 has dimension at least7. Set u_r=c_r*1+v_r with arbitrary relation-dependent c_r. Then alpha_r dot u_r=uniform dot u_r=c_r. Uniform rank-one routing exactly reproduces every logit b_r+c_r despite rank-nine native attention.

These utilities are realizable in the model's64D value/readout by embedding nine independent token values and choosing decoder rows with these coefficients. Nine independent1024D token inputs permit simultaneous finite-set interpolation of preparation angles and values by their separate linear maps. This is an existence counterexample, not an observed benchmark case or a utility result.

4. Exact dual sign fixture

K=1, A=(1,1)^T, U=[[0,1],[1,0]], z=(.9,.9). M has columns(-.9,.1)^T and(.1,-.9)^T. y=(-1,-1) gives delta=.8, output error lower bound.4. x=(.5,.5) achieves f=(.5,.5), attaining the bound exactly.

5. Consequences and unresolved evidence

Raw attention rank cannot establish task-relevant quantum expressivity. The conditional LP offers a falsifiable measurement with fixed actual utilities. A future diagnostic could freeze a native completed checkpoint, label-blind passage IDs and common query frame, then compare K1 vsK8 matches/certified separations/unresolved cases, retaining all timeouts. Three allocated3060 GPUs can handle disjoint native-forward shards while CPU solves LPs. This note launches no run.

No eligible L5 candidate is selected. A task-loss witness against all shared eight-component frames, nontrivial novel theory, and independent empirical support remain absent. The old L4 applies only to its original utility/local-alignment conjunction. New-method efficacy, phase necessity, contextuality and computational advantage remain unestablished.
