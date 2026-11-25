# Signed-value effect update: guarantees, controls and obstruction

This is an audit of the previously frozen expression, using known spectral and Rayleigh calculus. It is not a new theorem claim or a selected task method. All task values, keys and biases are fixed in the assertions below.

For valid pure token densities rho_t, put B_t=rho_t+epsilon I, B=sum B_t, A_c=sum u_ct B_t. For a unit class effect phi, the original native pooled score excluding bias is z0=phi* A phi/(phi* B phi). Here B is strictly positive since epsilon=1e-6 and at least one token is valid. Define C=A-z0 B, so phi* C phi=0, and q_s proportional to exp(sC)phi, for symbolic s>=0. C remains fixed at its original z0, rather than being recomputed during a differential flow.

## What the update proves

Let w_s=exp(sC)phi. Then z(q_s)-z0 = N(s)/D(s), with N=phi* C exp(2sC)phi and D=w_s* B w_s>0. Diagonalizing Hermitian C gives N(0)=0 and N'(s)=2 sum_j |a_j|^2 lambda_j^2 exp(2s lambda_j)>=0. Thus z(q_s)>=z0 for every finite s>=0. The initial derivative is 2||Cphi||^2/(phi* B phi). If Cphi is nonzero, then z(q_s)>z0 for every finite s>0. This is a strict comparison with the starting score, not monotonicity in s. Normalization of q cancels from the quotient.

This proves improvement relative to the starting score only. It does not prove monotonicity as s varies, since D also changes. The six-token fixture has three |minus> densities, one |plus>, |0>, |1>, utilities (0,0,0,0,-1,1), and phi=|plus>. On their two-dimensional support B=[[3+6epsilon,-1],[-1,3+6epsilon]], A=C=diag(-1,1), z0=0. With x=exp(2s), z=(x^2-1)/[(3+6epsilon)(x^2+1)-2x]. The exact rational values at x=6 and x=100 are positive but the second is smaller. The other six dimensions have B=6epsilon I and zero amplitudes. No task time is selected by this example.

Every updated row still uses nonnegative weights (q* rho_t q+epsilon)/sum_t(...), so its score stays in the original utility convex hull. An input-dependent free effect changes a shared constraint but does not remove the individual Rayleigh upper bound.

## Classification competition

All forty rows rising relative to themselves has no required sign for a classification margin. The CE derivative is sum_c (softmax(z)_c-1[c=y]) z'_c; the terms have different signs. A fixed forty-class counterexample uses two orthogonal word states and phi=|plus> for all rows. The correct row has utilities (1,1), a rival has (-2,2), and the other38 have (-100,-100), with zero bias. Initially the correct score1 beats rival0. At algebra s=1 the rival score is 2 tanh(4)/(1+2epsilon)>1, while every constant row is unchanged. CE increases and the prediction becomes incorrect, despite every row satisfying baseline nondecrease.

The retained shared-query diagnostic uses exactly the same checkpoint/state and original A/B convention. It reports431 original quantum errors with a negative numerical upper witness for ANY gold-row query relative to the fixed native top rival. Since the present update leaves that upper bound fixed and each rival score is at least its native value, those431 remain wrong under the fixed-teacher update. This is a conditional inference from previously computed FP64 certificates, not a new task replay or an exact-arithmetic certificate. It does not constrain jointly retrained keys/values/biases.

## Required classical controls

For M=X+iY and v=a+ib, the exact realification R(M)=[[X,-Y],[Y,X]], r(v)=(a,b) satisfies r(Mv)=R(M)r(v), preserves real quadratic forms, and intertwines matrix exponentials. Thus this entire update has an exact constrained 16-real-coordinate representation of its8-complex-coordinate computation. This supplies an implementation control, not a claim of equal capacity to all separately trained real8/real12 models or a veto of quantum research.

The strong scalar control alpha_tau(t) proportional to alpha0(t) exp(tau u_t) has score derivative Var_alpha_tau(u)>=0 for all tau. It already supplies baseline-monotone utility improvement. Operator filtering needs a task-linked phase/geometry prediction and fair comparisons beyond that property. Diagonal population updates and independently trained strong Gaussian/softmax controls remain necessary. Different u-driven nonunitary generators distinguish this expression from the exact gen90 key-only unitary, but that difference alone does not establish substantive novelty.
