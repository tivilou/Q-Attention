"""Exact finite-difference and stale-gradient witnesses for the global soft objective."""
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import sys


def counts(probs, labels, weights):
    nclass = len(probs[0])
    gold = [F(0)] * nclass
    pred = gold.copy()
    correct = gold.copy()
    for row, label, weight in zip(probs, labels, weights):
        assert sum(row) == 1 and min(row) >= 0
        gold[label] += weight
        correct[label] += weight * row[label]
        for i, v in enumerate(row):
            pred[i] += weight * v
    return [a+b for a,b in zip(gold,pred)], correct


def score(den, correct):
    assert min(den)>0
    return sum(2*x/y for x,y in zip(correct,den))/len(den)


def linearized_terms(den, correct, d_den, d_correct):
    return [2*(dt/d-t*dd/d**2) for d,t,dd,dt in zip(den,correct,d_den,d_correct)]


def difference(a,b):
    return [x-y for x,y in zip(a,b)]


def at(s):
    # Only two class coordinates vary; the other 38 have fixed perfect
    # one-row predictions. This is precisely a 40-class probability witness.
    a,b=F(1,10)+s/4,F(1,10)+s/10
    v=[[1-a,a],[1-b,b]]
    rows=[row+[F(0)]*38 for row in v]
    rows += [[F(i==j) for i in range(40)] for j in range(2,40)]
    return rows


def finite_certificate(den, true, next_den, next_true):
    lin=linearized_terms(den,true,difference(next_den,den),difference(next_true,true))
    exact=sum(g*d/e for g,d,e in zip(lin,den,next_den))/len(den)
    measured=score(next_den,next_true)-score(den,true)
    assert exact==measured
    drift=max(abs(e-d)/d for d,e in zip(den,next_den))
    assert drift<1
    approximate=sum(lin)/len(den)
    error_bound=drift/(1-drift)*sum(map(abs,lin))/len(den)
    assert abs(exact-approximate)<=error_bound
    return measured,approximate,drift,error_bound


def additive_grad(prob, labels, w):
    # Two coefficients per class, using only FIT predictions as anchors.
    den,t=counts(prob,labels,w)
    positive=[F(2,40)/d for d in den]
    tax=[F(2,40)*a/d**2 for a,d in zip(t,den)]
    def anchor_row(row,y,weight):
        return weight*(positive[y]*row[y]-sum(k*p for k,p in zip(tax,row)))
    return anchor_row


def prediction(p):
    return [max(range(len(row)),key=lambda k:row[k]) for row in p]


def comp():
    labels=list(range(40))
    weights=[F(1)]*40
    zero,one=at(F(0)),at(F(1))
    den,true=counts(zero,labels,weights)
    next_den,next_true=counts(one,labels,weights)
    d_den,d_correct=difference(next_den,den),difference(next_true,true)
    anchor_terms=linearized_terms(den,true,d_den,d_correct)
    anchor=sum(anchor_terms)/40
    anchor_row=additive_grad(zero,labels,weights)
    additive=sum(anchor_row(q,y,w)-anchor_row(p,y,w)
                 for p,q,y,w in zip(zero,one,labels,weights))
    assert additive==anchor and anchor>0
    assert prediction(zero)==prediction(one)
    rows=[]
    for t in range(11):
        f=F(t,10)
        current_den,current_true=counts(at(f),labels,weights)
        gradient=sum(linearized_terms(current_den,current_true,d_den,d_correct))/40
        delta,surrogate,drift,bound=finite_certificate(den,true,current_den,current_true)
        rows.append(dict(s=str(f),exact_direction=str(gradient),direction=float(gradient),
                         anchored_change=float(surrogate),global_change=float(delta),
                         max_relative_denominator_drift=float(drift),exact_error_bound=float(bound)))
    first=next(row for row in rows if row['direction']<0)
    assert first['s']=='3/10'
    assert rows[-1]['anchored_change']>0>rows[-1]['global_change']
    exact,approximate,drift,bound=finite_certificate(den,true,next_den,next_true)
    # A known fixed-prediction postprocessor cannot infer a soft-objective gain
    # from identical hard argmaxes along this path.
    return dict(status='PASS_EXACT_STALE_ANCHOR_REVERSAL_AND_FINITE_ERROR_CERTIFICATE',
                anchor_gradient=str(anchor),exact_final_change=str(exact),
                linearized_final_change=str(approximate),first_negative_point_on_fixed_grid=first,
                stale_steps=rows,all_40_hard_argmax_unchanged=True,
                bounds_verified_for_all_grid_points=True,per_sample_additive_equality=True,
                synthetic_record_count=40,task_forwards=0,optimizer_updates=0,gpu_allocations=0)


def budget(path):
    raw=Path(path).read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest()=='0b2da30bf1c0ca6b8809220c2fa72b598835c6e37c84b40c3c610ee4281e7e25'
    p=json.loads(raw)
    b=p['budget']
    step=p['training']['logical_batch']
    fit=p['data']['sizes']['fit']
    updates=p['training']['max_updates_per_arm']
    assert 4*updates*step==b['max_scientific_train_exposures']
    assert b['max_total_train_exposures']==b['max_scientific_train_exposures']+4*step
    scenarios=[]
    for passes in (1,4,20):
        extra=4*passes*fit
        scenarios.append(dict(full_fixed_snapshot_censuses_per_arm=passes,additional_fit_forwards=extra,
                              full_logical_batches_synchronized_all_arms=extra//step,
                              percent_of_frozen_scientific_update_exposures=100*extra/b['max_scientific_train_exposures'],
                              total_if_fit_read_only_forwards_charge_same_exposure_cap=
                              b['max_total_train_exposures']+extra))
    return dict(status='SOURCE_BOUND_FORWARD_ACCOUNTING_NO_TIMING_ASSUMPTION',
        work=b['max_scientific_train_exposures'],total=b['max_total_train_exposures'],scenarios=scenarios,
        protected_roles_used_as_anchors=0,
        scope='Census forwards are additional work at unchanged update counts. Actual execution, resource timing and exact cap classification are unverified.')


if __name__=='__main__':
    print(json.dumps(dict(exact=comp(),budget=budget(sys.argv[1])),indent=2,allow_nan=False))