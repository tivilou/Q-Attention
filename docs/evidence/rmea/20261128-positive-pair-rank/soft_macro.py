"""Exact directional counterexample and aggregate-gradient check for soft Macro-F1."""
from fractions import Fraction as F
import itertools
import json


def moment(prob, gold, weights):
    k = len(prob[0])
    support = [sum(w for y,w in zip(gold,weights) if y==c) for c in range(k)]
    predicted = [sum(w*p[c] for w,p in zip(weights,prob)) for c in range(k)]
    true = [sum(w*p[c] for y,w,p in zip(gold,weights,prob) if y==c) for c in range(k)]
    return [a+b for a,b in zip(support,predicted)], true


def macro(prob, gold, weights):
    den, tp = moment(prob,gold,weights)
    return sum(2*t/d if d else F(0) for t,d in zip(tp,den))/len(den)


def coefficient(prob, gold, weights):
    den, tp = moment(prob,gold,weights)
    def cell(c, label, weight, total, hit):
        return 2*weight*(F(label==c)/total-hit/total**2)/len(den) if total else F(0)
    return [[cell(c,y,w,d,t) for c,(d,t) in enumerate(zip(den,tp))]
            for y,w in zip(gold,weights)]


def matrix_dot(grad, v):
    return sum(a*b for x,y in zip(grad,v) for a,b in zip(x,y))


def move(prob, direction, step):
    return [[a+b*step for a,b in zip(x,y)] for x,y in zip(prob,direction)]


def work():
    probs=[[F(4,5),F(1,5)],[F(1,4),F(3,4)]]
    labels=[0,1]
    weights=[F(1),F(1)]
    slope=[[-F(1,10),F(1,10)],[-F(12,125),F(12,125)]]

    def global_fun(x):return macro(x,labels,weights)
    def batch_fun(x):return sum(macro([p],[y],[w]) for p,y,w in zip(x,labels,weights))/2

    exact=matrix_dot(coefficient(probs,labels,weights),slope)
    average=sum(matrix_dot(coefficient([p],[y],[w]),[v])
                for p,y,w,v in zip(probs,labels,weights,slope))/2
    step=F(1,100000)
    for fn,derivative in [(global_fun,exact),(batch_fun,average)]:
        df=(fn(move(probs,slope,step))-fn(move(probs,slope,-step)))/(2*step)
        assert abs(df-derivative)<F(1,100000000)
    assert exact<0<average
    assert global_fun(move(probs,slope,F(1,100)))<global_fun(probs)
    assert batch_fun(move(probs,slope,F(1,100)))>batch_fun(probs)

    # At a frozen anchor, uniform rows estimate the gradient of the
    # linearization exactly after N/B scaling, including fixed row weights.
    p=[[F(5,8),F(1,8),F(2,8)],[F(1,8),F(4,8),F(3,8)],
       [F(3,8),F(1,8),F(4,8)],[F(2,8),F(3,8),F(3,8)]]
    y=[0,1,2,2]
    w=[F(2),F(1,3),F(1,2),F(3)]
    s=[[-F(2,100),F(1,100),F(1,100)],[F(1,100),F(0),-F(1,100)],
       [-F(1,100),-F(1,100),F(2,100)],[F(1,100),-F(1,100),F(0)]]
    g=coefficient(p,y,w)
    gradient=matrix_dot(g,s)
    subsets=list(itertools.combinations(range(4),2))
    estimates=[2*matrix_dot([g[i] for i in ix],[s[i] for i in ix]) for ix in subsets]
    assert gradient==sum(estimates)/len(estimates)
    finite=(macro(move(p,s,step),y,w)-macro(move(p,s,-step),y,w))/(2*step)
    assert abs(finite-gradient)<F(1,100000000)
    return dict(status='PASS_EXACT_SOFT_MACRO_BATCH_DIRECTION_COUNTEREXAMPLE',
                full_direction=str(exact),mean_batch_direction=str(average),
                full_approx=float(exact),mean_batch_approx=float(average),
                all_four_choose_two_matched=True,weighted_anchor_direction=str(gradient),
                scope='Artificial probabilities, exact local gradient. Not hard Macro-F1, stale-anchor validation, model novelty, or measured gain.',
                task_forward=0,optimizer_updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(work(),indent=2,allow_nan=False))