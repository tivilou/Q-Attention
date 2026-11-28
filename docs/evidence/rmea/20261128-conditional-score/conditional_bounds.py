"""Argmax certificates for bounded native probabilities, not a training shortcut."""

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


EPS = 1e-6
TOL = 1e-11  # Fixture tolerance, not a production or device error certificate.


def extremal_means(values, lower, upper):
    """Tight vectorized scalar endpoints for independent token-weight boxes."""
    assert values.ndim == 2 and values.shape == lower.shape == upper.shape
    assert values.shape[1] > 0 and np.all(lower > 0) and np.all(upper >= lower)
    # At an optimum, switch to upper weights only below (min) or above (max)
    # the attained mean. Equal values can use either weight, so n+1 corners suffice.
    order = np.argsort(values, axis=1, kind='stable')
    u, low, high = (np.take_along_axis(a, order, axis=1) for a in (values, lower, upper))

    def sorted_extreme(v, l, h, maximum):
        delta = h-l
        ps = np.concatenate((np.zeros((v.shape[0], 1)), np.cumsum(delta*v, axis=1)), axis=1)
        pw = np.concatenate((np.zeros((v.shape[0], 1)), np.cumsum(delta, axis=1)), axis=1)
        ratios = ((l*v).sum(axis=1, keepdims=True)+ps)/(l.sum(axis=1, keepdims=True)+pw)
        return ratios.max(axis=1) if maximum else ratios.min(axis=1)

    return sorted_extreme(u, low, high, False), sorted_extreme(u[:, ::-1], low[:, ::-1], high[:, ::-1], True)


def candidates(low, high):
    # numpy.argmax's first-index tie rule. Only strict inequalities certify removal
    # here; retaining the additional exact-tie classes is intentional.
    return np.flatnonzero(high+TOL >= low.max()-TOL)


def q_prob(queries, words, mode):
    if mode == 'Q':
        return np.abs(queries.conj()@words.T)**2
    return np.abs(queries)**2@np.abs(words.T)**2


def q_box(queries, words, k, mode):
    q, w = queries[:, k:], words[:, k:]
    if mode == 'Q':
        prefix = np.abs(queries[:, :k].conj()@words[:, :k].T)
        radius = np.linalg.norm(q, axis=1)[:, None]*np.linalg.norm(w, axis=1)[None, :]
        low = np.maximum(0., prefix-radius)**2
        high = np.minimum(1., (prefix+radius)**2)
    else:
        low = np.abs(queries[:, :k])**2@np.abs(words[:, :k].T)**2
        high = low + np.max(np.abs(q)**2, axis=1)[:, None]*np.sum(np.abs(w)**2, axis=1)[None, :]
        high = np.minimum(1., high)
    # Widen bounds on this floating-point fixture; real arithmetic has no slack.
    return np.maximum(0., low-TOL)+EPS, np.maximum(low, high+TOL)+EPS


def gaussian_box(queries, words, k):
    q, w = queries[:, k:], words[:, k:]
    base = ((queries[:, None, :k]-words[None, :, :k])**2).sum(-1)
    qn, wn = np.linalg.norm(q, axis=1)[:, None], np.linalg.norm(w, axis=1)[None, :]
    low = np.exp(-base-(qn+wn)**2)
    high = np.exp(-base-(qn-wn)**2)
    return np.maximum(0., low-TOL)+EPS, high+TOL+EPS


def gaussian_covariance_bound(values, linear, covariance, bias, low, high):
    # Bound the complete strong classical score, not merely its mean branch.
    # For each paired coordinate, bound E[x*y] and E[x]E[y] separately.
    vl, vu = [], []
    for v in (values, values[:, 0::2]*values[:, 1::2]):
        l, u = zip(*(extremal_means(np.broadcast_to(v[:, i], low.shape), low, high) for i in range(v.shape[1])))
        vl.append(np.stack(l, axis=1)); vu.append(np.stack(u, axis=1))
    l, u = vl[0], vu[0]
    products = np.stack([a[:, 0::2]*b[:, 1::2] for a,b in itertools.product((l,u),repeat=2)])
    cl = vl[1]-products.max(0)
    cu = vu[1]-products.min(0)
    # Covariance is also at most one-quarter of the product of the observed ranges.
    cap = np.ptp(values[:, 0::2],axis=0)*np.ptp(values[:, 1::2],axis=0)/4
    cl, cu = np.maximum(cl,-cap), np.minimum(cu,cap)
    ul, uu = extremal_means(linear,low,high)
    return (ul+bias+np.minimum(covariance*cl,covariance*cu).sum(1),
            uu+bias+np.maximum(covariance*cl,covariance*cu).sum(1))


def corner_fixture():
    rng = np.random.default_rng(121)
    for n in range(1,8):
        for _ in range(20):
            v = rng.integers(-5,6,(3,n))/3
            lo = rng.uniform(.01,1,(3,n))
            hi = lo+rng.uniform(0,1,(3,n))
            scores = []
            for corner in itertools.product((False,True),repeat=n):
                weights = np.where(corner,hi,lo)
                scores.append((weights*v).sum(1)/weights.sum(1))
            l,u = extremal_means(v,lo,hi)
            assert np.allclose(l,np.min(scores,axis=0),rtol=0,atol=TOL)
            assert np.allclose(u,np.max(scores,axis=0),rtol=0,atol=TOL)
    # A class can have the same winning decision while loser scores, CE,
    # and a comparison involving a fixed-weight ablation all differ.
    a = np.array([2.,1.,0.]); b = np.array([2.,-10.,-10.])
    ce = lambda x: float(np.log(np.exp(x-x.max()).sum())+x.max()-x[0])
    assert a.argmax()==b.argmax()==0 and ce(a)>ce(b)+.4
    return dict(box_corner_groups=140, ce_same_prediction=[ce(a),ce(b)])


def audit():
    rng = np.random.default_rng(20261128)
    ret = {k:dict(full_dimension=d, initial_bound_dimensions=d//2, eligible_counts=[],
                  every_true_winner_retained=True, each_bound_contains_all_40=True, full_endpoint_error=0.)
           for k,d in [('Q',8),('D',8),('Gcov',12)]}
    for n in (1, 4, 12, 40):
        for _ in range(8):
            def unit(a):return a/np.linalg.norm(a,axis=1)[:, None]
            w = unit(rng.normal(size=(n,8))+1j*rng.normal(size=(n,8)))
            q = unit(rng.normal(size=(40,8))+1j*rng.normal(size=(40,8)))
            qw,ww = rng.normal(size=(40,12))*.2,rng.normal(size=(n,12))*.2
            v = rng.normal(size=(n,64))
            decoder = rng.normal(size=(40,64))/8
            bias = rng.normal(size=40)*.1
            covariance = rng.normal(size=(40,32))/8
            utilities = decoder@v.T
            for mode, stats in ret.items():
                if mode=='Gcov':
                    weight = np.exp(-((qw[:, None]-ww[None])**2).sum(-1))+EPS
                    box = lambda k:gaussian_box(qw,ww,k)
                else:
                    weight = q_prob(q,w,mode)+EPS
                    box = lambda k:q_box(q,w,k,mode)
                alpha = weight/weight.sum(1,keepdims=True)
                logits = (alpha*utilities).sum(1)+bias
                if mode=='Gcov':
                    mean = alpha@v
                    conn = alpha@(v[:,0::2]*v[:,1::2])-mean[:,0::2]*mean[:,1::2]
                    logits += (conn*covariance).sum(1)
                lo,hi = box(stats['initial_bound_dimensions'])
                assert np.all(lo <= weight+TOL) and np.all(weight <= hi+TOL)
                def bound(low,high):
                    if mode=='Gcov':return gaussian_covariance_bound(v,utilities,covariance,bias,low,high)
                    l,u = extremal_means(utilities,low,high)
                    return l+bias,u+bias
                a,b = bound(lo,hi)
                assert np.all(a<=logits+TOL) and np.all(logits<=b+TOL)
                alive = candidates(a,b)
                assert logits.argmax() in alive and alive[logits[alive].argmax()]==logits.argmax()
                stats['eligible_counts'].append(int(len(alive)))
                e,f = bound(weight,weight)
                err = max(abs(e-logits).max(),abs(f-logits).max())
                assert err < TOL
                stats['full_endpoint_error'] = max(stats['full_endpoint_error'],float(err))
    for mode, stats in ret.items():
        rest = stats['full_dimension']-stats['initial_bound_dimensions']
        # Indexed data passes, not GPU elapsed, arithmetic operation or call counts.
        counts = stats.pop('eligible_counts')
        stats.update(fixture_mean_candidate_count=float(np.mean(counts)),
                     full_40_fixture_rows=sum(n==40 for n in counts),
                     exits_before_full_contraction=sum(n<40 for n in counts),
                     average_class_token_coordinate_lanes_unweighted_by_length=float(stats['initial_bound_dimensions']*40+rest*np.mean(counts)),
                     dense_class_coordinate_lanes=40*stats['full_dimension'],
                     all_40_class_values_used_for_bound=True)
    return ret


def run(native_path, gcov_path):
    def pin(path, fragment):
        raw = Path(path).read_bytes()
        assert fragment.encode() in raw
        return hashlib.sha256(raw).hexdigest()
    native_pin=pin(native_path,'logits=(alpha*utilities).sum(-1)+self.bias')
    assert native_pin=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
    gcov_pin=pin(gcov_path,'(connected*self.moment_decoder).sum(-1)')
    return dict(status='PASS_LABEL_FREE_ARGMAX_CERTIFICATE_BUT_NOT_ALL_SCORE_EVIDENCE',
                fixture=corner_fixture(),synthetic_records=32,arms=audit(),native_source_sha256=native_pin,
                gcov_source_sha256=gcov_pin,all_64_value_coordinates=True,
                scope='Artificial CPU inputs and exact real-arithmetic bounds. No GPU timing or claim of reduced circuit/state preparations.',
                task_forwards=0,updates=0,gpu_allocations=0)


if __name__ == '__main__':
    print(json.dumps(run(*sys.argv[1:]),indent=2,allow_nan=False))
