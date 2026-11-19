# Finite cross-fold classification-risk window and its argmax ceiling

Status: primary analytic derivation; independent review is tracked separately. This
extends classification-risk-v001.md without changing the model, training
contract, noise distribution, parameters, experiment gates or prior evidence.
It is one technical lemma for the same mechanism, not another idea or L5.

## Frozen interface and assumptions

Reuse v001 Sections1-2 exactly: d=8,C=40,EPS=1e-6,eta=1,zero biases,
phi=(e0+e1)/sqrt(2) for every class, four states(e0,e0,e1,e1),alternating
A={0,2},B={1,3}, native token weights1/4 and fold masses1/2. True class y=7
has coherent amplitudes(s,s). Every distractor has amplitudes(xi*t,zeta*t),
with the same independent uniform signs xi,zeta across all39 distractors.
This is a specified interface distribution. Neither latent NLP error
identification nor original feature/circuit reachability has been established.

Put

    k = log((1+EPS)/EPS)/2, q = tanh(k)/k, p = 39/40,
    Ck = 1 + 1/(2*k^2),
    a(b) = b/sqrt(1+2*b^2),
    h(x) = x*tanh(sqrt(k^2+x^2))/sqrt(k^2+x^2),
    F(b) = b*h(a(b)),
    R(b) = b*h(2*b/sqrt(1+8*b^2)),
    L(x) = log(1+39*exp(x)).

The exact expected per-draw CE risks from v001, not averaged-logit CE, are

    N = L(0), S = L(F(t)-F(s)),
    X = [L(F(t)-F(s)) + L(-F(t)-F(s))]/2,
    U = [L(R(t)-R(s)) + L(-R(s))]/2,

for native,self_fold,cross_fold,unpartitioned full respectively.

## 1. Global elementary remainder bounds

For every b>=0,

    0 <= q*b^2 - F(b) <= Ck*q*b^4,                    (1)
    0 <= 2*q*b^2 - R(b) <= 8*Ck*q*b^4.               (2)

Proof. The function g(T)=tanh(T)/T is decreasing for T>0 because
T*sech(T)^2-tanh(T)<0. For T=sqrt(k^2+x^2),

    1 >= g(T)/g(k) >= k/T = (1+x^2/k^2)^(-1/2).

For A=(1+2*b^2)^(-1/2) and B=(1+a(b)^2/k^2)^(-1/2),
F(b)/(q*b^2)>=A*B and is at most1 when b>0. Since

    1-A*B <= (1-A)+(1-B)
            <= b^2 + a(b)^2/(2*k^2) <= Ck*b^2,

equation(1) follows; b=0 is immediate. For R, replace A by
(1+8*b^2)^(-1/2) and x by2*b/sqrt(1+8*b^2). The two deficits are at most
4*b^2 and2*b^2/k^2; multiplying their sum by2*q*b^2 proves(2).
The inequality1-(1+v)^(-1/2)<=v/2 holds for all v>=0, so these are global
bounds even where their lower estimates are uninformative.

Also, for all real x,

    0 <= L(x)-L(0)-p*x <= x^2/8,                     (3)

because L'(0)=p and0<L''(x)<=1/4. For x>=0,

    h'(x) = k^2*tanh(T)/T^3 + x^2*sech(T)^2/T^2 > 0,
    T=sqrt(k^2+x^2).

Thus F and R are strictly increasing on b>0. In particular, for t>s>0,
0<R(t)-R(s)<=R(t)<=2*q*t^2.

## 2. Explicit finite advantage region

Let r=t/s with1<r<sqrt(2), and define

    B1 = Ck*(4*r^4+1) + q*(1+r^4)/(8*p),
    B2 = 8*Ck + q*(1+r^4)/(4*p).

Then the same risks obey

    U-X >= p*q*s^2 * [(r^2-1)-B1*s^2],              (4)
    N-U >= p*q*s^2 * [(2-r^2)-B2*s^2].              (5)

Consequently the explicit nonempty interval

    0 < s^2 < min((r^2-1)/B1,(2-r^2)/B2)            (6)

implies

    X < U < N < S.                                  (7)

Proof of the quantitative bounds. By(3), with Fs=F(s),Ft=F(t),

    X <= L(0)-p*q*s^2 + p*Ck*q*s^4
                         + q^2*(s^4+t^4)/8.

Write R(b)=2*q*b^2-e_b with0<=e_b<=8*Ck*q*b^4.
The mean argument in U is R(t)/2-R(s), hence(3) gives

    U >= L(0)+p*q*(t^2-2*s^2)-4*p*Ck*q*t^4,
    U <= L(0)+p*q*(t^2-2*s^2)+8*p*Ck*q*s^4
                         +q^2*(s^4+t^4)/4.

The upper second-order term uses the average of the squared arguments
(R(t)-R(s))^2 and R(s)^2, bounded by2*q^2*(t^4+s^4).
Subtract the bounds to obtain(4)-(5). Finally Ft>Fs gives S>N, while
strict monotonicity of L gives X<S for all s,t>0. No numeric threshold
search or asymptotic remainder is used in(4)-(7).

A simple closed instance uses the original EPS, for which k>1. Choose
r=sqrt(3/2). Since q<=1,Ck<=3/2 and p=39/40,

    B1 <= 15+5/12 = 185/12 < 16,
    B2 <= 12+5/6  = 77/6   < 13.

Therefore every0<s<=1/8 satisfies(7), with the explicit gaps

    U-X >= (199/768)*p*q*s^2 > p*q*s^2/4,
    N-U >= (115/384)*p*q*s^2 > p*q*s^2/4.            (8)

This is a conservative finite sufficient window, not the sharp boundary or
an observed task regime. r and s specify the analytic input law; no algorithm
coefficient, optimizer or training contract has been changed.

## 3. The CE witness does not distinguish cross/full argmax accuracy

For every t>s>0 under the same interface distribution, self_fold always
ranks distractors above the true class. For cross_fold, the true logit Fs>0
beats all distractors exactly when xi*zeta=-1; if xi*zeta=1, every distractor
has logit Ft>Fs and the true class loses. For full, the true logit Rs>0
beats all distractors exactly when xi!=zeta; if xi=zeta the distractor
logit Rt>Rs and the true class loses. These are the identical sign events.
Therefore

    P(argmax cross == y) = P(argmax full == y) = 1/2,
    P(argmax self == y) = 0.                            (9)

The correct/incorrect event equality is independent of how distractor ties
are broken, since the true class is strictly above or below them. Native's
all-zero tie must be stated separately: under the executed lowest-index
argmax and y=7 it is always incorrect. Another tie rule could differ.

More generally cross and full need not make identical wrong-class choices
under an unspecified tie policy. With the same deterministic lowest-index
rule, their complete argmax predictions coincide on every draw in this
witness. Replicating draws under that rule therefore cannot create a
cross-versus-full confusion-matrix or positive39-Macro difference.
Equation(8) concerns finite confidence/loss behavior, not more correct
classifications versus full. It supplies no theorem that the frozen real
training Macro gate will pass.

## Claim ceiling and next observation

The finite CE ranking is a source-interface technical prediction. The argmax
ceiling prevents using it as a proof of Macro improvement. Real jointly fitted
representations, class-dependent errors, finite noisy observations and held-out
task benefits are still unresolved. Noise2Self's known coordinate-isolation
principle remains a near neighbor; this calculation is not novelty clearance.

No code, task, coefficient, gate, formal run or storage contract is altered.
The exact reviewed source815cfb2 training retains priority when the shared
scheduler's global-quiescence gate permits its single attempt. New independent
full-data confirmation remains collaborator-only. L5 NOT_ACCEPTED.
