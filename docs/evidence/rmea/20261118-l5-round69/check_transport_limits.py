"""Exact arithmetic certificates; no model import, tensor data or training."""
from fractions import Fraction as F
import json


def winner(scores):return max(range(len(scores)),key=lambda k:scores[k])


def main():
    eps=F(1,10**6);R=40;T=2
    kernel=[[1+eps,eps] for _ in range(R)]
    original=[[q/sum(row) for q in row] for row in kernel]
    # Closed-form unique positive balanced solution for identical kernel rows.
    plan=[[F(1,R*T)]*T for _ in range(R)]
    assert all(sum(row)==F(1,R) for row in plan)
    assert all(sum(plan[r][t] for r in range(R))==F(1,T) for t in range(T))
    a=[F(1,R*T)/(1+eps)]*R;b=[F(1),(1+eps)/eps]
    assert all(plan[r][t]==a[r]*kernel[r][t]*b[t] for r in range(R) for t in range(T))
    balanced=[[R*q for q in row] for row in plan]
    assert all(sum(row)==1 for row in balanced)
    # Same legal rank-one Born frame. Opposite signed readout utility is allowed.
    def scores(weights,useful_token,null_bias):
        result=[F(-1)]*R;result[0]=null_bias;result[1]=weights[1][useful_token]
        return result
    harm_before=scores(original,0,F(3,4));harm_after=scores(balanced,0,F(3,4))
    gain_before=scores(original,1,F(1,4));gain_after=scores(balanced,1,F(1,4))
    assert winner(harm_before)==1 and winner(harm_after)==0
    assert winner(gain_before)==0 and winner(gain_after)==1
    # Diagonal scaling preserves every multiplicative 2x2 cross-ratio.
    K=[[F(2),F(3)],[F(5),F(7)]];aa=[F(11),F(13)];bb=[F(17),F(19)]
    W=[[aa[r]*K[r][t]*bb[t] for t in range(2)] for r in range(2)]
    assert W[0][0]*W[1][1]/(W[0][1]*W[1][0])==K[0][0]*K[1][1]/(K[0][1]*K[1][0])
    # Row conditioning cancels the row multiplier, retaining a shared token prior.
    conditioned=[[x/sum(row) for x in row] for row in W]
    explicit=[[K[r][t]*bb[t]/sum(K[r][u]*bb[u] for u in range(2)) for t in range(2)] for r in range(2)]
    assert conditioned==explicit
    out=dict(status='PASS_EXACT_TRANSPORT_LIMITS',epsilon=str(eps),rows=R,tokens=T,
        original_weights=[str(x) for x in original[0]],balanced_weights=[str(x) for x in balanced[0]],
        harmful=dict(before_winner=winner(harm_before),after_winner=winner(harm_after),before_positive=str(harm_before[1]),after_positive=str(harm_after[1])),
        helpful=dict(before_winner=winner(gain_before),after_winner=winner(gain_after),before_positive=str(gain_before[1]),after_positive=str(gain_after[1])),
        rectangular_marginals_exact=True,diagonal_scaling_certificate_exact=True,crossratio_exact=True,
        theory_status='KNOWN_TOOLS_AND_SCOPED_COUNTEREXAMPLES_NOT_L5',model_forwards=0,training=False,L5='NOT_ACCEPTED')
    print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
