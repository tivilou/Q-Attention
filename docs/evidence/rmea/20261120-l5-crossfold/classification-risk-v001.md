# Cross-fold classification advantage and its reversal: source-interface witnesses

Primary derivation independently checked: seven mathematical checks PASS; see risk-review-record.json. This describes
the same cross_fold/self_fold/full maps, not a new mechanism family. No numerical
execution, actual NLP noise model, novelty, task utility or L5 is established.

## 1. Fixed source interface

Use d=8, C=40, EPS=1e-6, eta=1, and zero class biases. All queries equal
phi=(e0+e1)/sqrt(2). Four valid tokens have states (e0,e0,e1,e1).
Valid-order alternating folds A={0,2}, B={1,3} therefore each contain (e0,e1).
Native token weights are 1/4 and native fold masses are 1/2. Nothing below
changes the source normalization, adds a trainable tensor, detaches gradients,
or introduces a classical residual. Utilities, with states fixed, are inputs
to the measured interface. Original feature/circuit reachability is not proved.

For one class, a fold of amplitude b has utility pair (b,-b). Its native
aggregate is zero, covariance is A_D=b sigma_z on span(e0,e1), and
N_D=sqrt(1+2b^2). Put

    k = log((1+EPS)/EPS)/2 > 0,
    a(b) = b/sqrt(1+2b^2),
    h(x) = x*tanh(sqrt(k^2+x^2))/sqrt(k^2+x^2),
    F(b) = b*h(a(b)).

On that two-dimensional subspace log E0=c I+k sigma_x. A donor of amplitude b
therefore produces exp(c I+k sigma_x+a(b) sigma_z). Its recipient weight
difference, after normalization within (e0,e1), is exactly h(a(b)). The common
trace normalization, the unused six dimensions and exp(c) cancel. h is odd,
F is even, and F(b)>0 for b!=0.

If the two fold amplitudes are b_A,b_B, the exact class logits are

    self(b_A,b_B)  = [b_A h(a(b_A))+b_B h(a(b_B))]/2,
    cross(b_A,b_B) = [b_A h(a(b_B))+b_B h(a(b_A))]/2.       (1)

For the existing unpartitioned full update, two coherent folds of amplitude b
give A_full=2b sigma_z, N_full=sqrt(1+8b^2), hence define

    R(b) = b*h(2b/sqrt(1+8b^2)).                         (2)

Then full(b,b)=R(b), full(b,-b)=0. Native logits vanish in all these cases.

## 2. Positive classification region, including the existing full control

Fix the true class y=7. Its clean fold amplitudes are s,s with s>0 and no noise.
Each of the other 39 classes has clean utilities zero and noisy amplitudes
(xi*t,zeta*t), where xi,zeta are independent uniform signs and t>0. The same
two signs are shared by all 39 distractor classes. This correlation within a
fold is allowed; the complete error vectors of the two folds are independent
and each has conditional mean zero given the fixed states/clean utilities.
This is a stipulated synthetic distribution, not an identified NLP error model.

The true-class logit is F(s) for self/cross and R(s) for full. Every distractor
logit is F(t) for self, xi*zeta*F(t) for cross, and either R(t) or zero for full,
each with probability 1/2. Define L(x)=log(1+39 exp(x)). Expected *per-draw*
40-class CE, not CE of an averaged logit, is exactly

    risk_native = L(0),
    risk_self   = L(F(t)-F(s)),
    risk_cross  = [L(F(t)-F(s))+L(-F(t)-F(s))]/2,
    risk_full   = [L(R(t)-R(s))+L(-R(s))]/2.              (3)

In particular risk_cross<risk_self for every s,t>0, by strict monotonicity of L.
This is not a universal statement about arbitrary clean/noisy utility laws.

Let q=tanh(k)/k>0. As b tends to zero,

    F(b)=q b^2+O(b^4),   R(b)=2q b^2+O(b^4).

Fix any ratio r in the open interval (1,sqrt(2)), and set t=r*s. Since
L'(0)=39/40, expansion of the exact expected risks gives

    risk_cross  = L(0) - (39/40) q s^2 + O(s^4),
    risk_full   = L(0) + (39/40) q (r^2-2) s^2 + O(s^4),
    risk_self   = L(0) + (39/40) q (r^2-1) s^2 + O(s^4).

For every fixed r in that interval, there exists s0(r,EPS)>0 such that for all
0<s<s0 and t=r*s,

    risk_cross < risk_full < risk_native < risk_self.    (4)

Proof: h(x)=q*x+O(x^3); insert a(b)=b+O(b^3) and the full argument 2b+O(b^3).
All arguments of L are O(s^2), so its Taylor remainder is O(s^4). The leading
coefficients are strictly ordered -1 < r^2-2 < 0 < r^2-1. Their finite strict
gaps dominate the remainder at sufficiently small positive s. No finite numeric
threshold s0 or measured task advantage is claimed. This is an open
signal/noise parameter region, not a seed/algorithm-coefficient search.

The mechanism prediction is specific: coherent true-class evidence survives
cross construction, while independent distractor self-selection loses its
positive mean. Self_fold matches partitioning and two-effect compute; full
uses one effect and has a different smooth-normalization response. Equal
parameters do not imply equal runtime. A classical matched implementation may
realize the same map; no quantum-necessity or resource claim follows.

## 3. Independent noise can reverse the self/cross risk order

Use the same states, queries and fixed true class. Now all distractor utilities
are zero. The true class has clean amplitudes (s,s), but independent fold errors
are uniform signs times s, so observed amplitudes b_A,b_B independently equal
0 or 2s, with mean s. Conditional zero-mean and cross-fold independence hold.

Equation (1) gives cross logits zero when at least one fold is zero, and
F(2s) when both are 2s. Self gives zero when both are zero, F(2s)/2 when exactly
one is 2s, and F(2s) when both are 2s. Thus self>=cross pointwise, with a strict
inequality of positive probability. Since true-class CE is strictly decreasing
in this logit,

    E CE_self < E CE_cross.                              (5)

Cross's clean-value readout identity still holds for its *noisy donor routes*.
It does not equal the score under the clean-construction route. In fact,

    E z_cross = F(2s)/4 < F(s)=z_cross_clean.             (6)

For strict inequality, write g(b)=h(a(b)). For b>0,

    g(b)/b = [1/sqrt(1+2b^2)] * [tanh(T(b))/T(b)],
    T(b)=sqrt(k^2+a(b)^2).

The first factor strictly decreases. a(b) and T(b) strictly increase, and
tanh(T)/T strictly decreases for T>0: its derivative numerator is
T sech^2(T)-tanh(T)<0, as tanh(T)-T sech^2(T) starts at zero and has derivative
2T sech^2(T)tanh(T)>0. Both factors are positive, hence g(2s)<2g(s).
Multiplying by s/2 gives F(2s)/4 < F(s). Strictly convex decreasing CE and Jensen
also imply E CE_cross > CE_cross_clean.

## 4. Decision boundary and claim ceiling

These paired witnesses replace an uninformative universal-benefit expectation
with a discriminating hypothesis. Cross construction may help when nuisance
competition dominates self-selection but hurt when useful evidence is noisy
or confined to one fold. Merely splitting contextual tokens establishes neither
distribution. An empirical test must examine actual label-relevant/distractor
coherence and task CE/Macro under common fitting, with self_fold/full/native and
appropriate classical controls. Existing development samples are discovery
only; no independent confirmation, novelty clearance or formal handoff follows.

The current frozen eight-check source canary remains unchanged. Equations
(3)-(6) are new analytic predictions and have not been numerically executed;
they require a separately frozen validation extension before empirical claims.
