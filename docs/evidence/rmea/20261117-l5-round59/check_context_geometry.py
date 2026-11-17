"""Exact rational fixtures for context-conditioned Born attention geometry.

Known POVM/data-processing tools applied to the source normalization; no model,
dataset, optimizer, circuit-reachability or new-method experiment.
"""
from fractions import Fraction as F
from pathlib import Path
import json, math
P=Path(__file__).resolve().parent
EPS=F(1,10**6)
a,b=F(9999,10001),F(200,10001)
QUERIES=[(a,b),(a,-b)]
assert a*a+b*b==1
def dot(x,y):return sum(u*v for u,v in zip(x,y))
def alpha(query,keys):
    scores=[dot(query,key)**2+EPS for key in keys]
    return [s/sum(scores) for s in scores]
def ce(z,y):return math.log1p(sum(math.exp(float(v-z[y])) for r,v in enumerate(z) if r!=y))
def audit(keys):
    assert all(dot(k,k)==1 for k in keys)
    alphas=[alpha(q,keys) for q in QUERIES]
    # Both fixed contexts have diagonal B=sum_t(|key><key|+eps I8).
    assert sum(k[0]*k[1] for k in keys)==0
    d0=sum(k[0]**2 for k in keys)+len(keys)*EPS
    d1=sum(k[1]**2 for k in keys)+len(keys)*EPS
    inner=lambda x,y:x[0]*d0*y[0]+x[1]*d1*y[1]
    effective_fidelity=inner(*QUERIES)**2/(inner(QUERIES[0],QUERIES[0])*inner(QUERIES[1],QUERIES[1]))
    tv=sum(abs(x-y) for x,y in zip(*alphas))/2
    assert tv*tv<=1-effective_fidelity
    # A fixed arbitrary signed utility row; the TV duality bound is tight here.
    u=[F(3,2),F(-7,4)]
    diff=abs(dot(u,alphas[0])-dot(u,alphas[1]))
    assert diff==(max(u)-min(u))*tv
    # Full40 logits, arbitrary signed utility rows and biases, fixed physical
    # inputs and queries. A cyclic query reassignment alone is the intervention.
    utilities=[(F(r+1,40),-F(41-r,40)) for r in range(40)]
    biases=[-F(r,100) for r in range(40)]
    z=[dot(utilities[r],alphas[r%2])+biases[r] for r in range(40)]
    changed=[dot(utilities[r],alphas[(r+1)%2])+biases[r] for r in range(40)]
    delta=[(max(u)-min(u))*tv for u in utilities]
    assert all(abs(v-w)<=d for v,w,d in zip(z,changed,delta))
    pred=max(range(40),key=lambda r:z[r]);newpred=max(range(40),key=lambda r:changed[r])
    certified=all(z[pred]-z[r]>delta[pred]+delta[r] for r in range(40) if r!=pred)
    if certified:assert pred==newpred
    y=7 # Fixed before all outputs; not a selected successful target.
    upper=math.log1p(sum(math.exp(float(z[r]-z[y]+delta[r]+delta[y])) for r in range(40) if r!=y))
    assert ce(changed,y)<=upper+1e-12
    return dict(attention_tv=str(tv),effective_query_fidelity=str(effective_fidelity),
        illustrative_floats=dict(TV=float(tv),effective_fidelity=float(effective_fidelity)),
        exact_tv_data_processing_checked=True,signed_utility_TV_duality_checked=True,
        full40=dict(query_permutation='r -> (r+1) mod40',native_model=False,
           fixed_target=y,original_argmax=pred,reassigned_argmax=newpred,
           decision_certified=certified,CE_original=ce(z,y),CE_reassigned=ce(changed,y),CE_upper=upper))
def main():
    contexts={'uninformative':[(F(1),F(0)),(F(1),F(0))],
              'discriminating':[(b,a),(b,-a)]}
    result={k:audit(v) for k,v in contexts.items()}
    assert F(result['uninformative']['attention_tv'])==0
    assert F(result['discriminating']['attention_tv'])>F(99,100)
    raw=dot(*QUERIES)**2;assert raw>F(998,1000)
    output=dict(schema='qattn.context-conditioned-query-audit.v1',status='PASS',epsilon=str(EPS),
        exact_arithmetic='Fraction for all normalization, fidelity, TV and logit quantities; floats only for illustrative log-exp evaluation.',
        fixture_dimension='Two-dimensional support embedded in C8; no native depth2 ansatz reachability claim.',
        fixed_query_coordinates=[[str(x) for x in q] for q in QUERIES],raw_query_fidelity=str(raw),
        raw_query_fidelity_float=float(raw),contexts=result,
        conclusion='The same fixed nearly identical queries yield TV0 or TV>.99 in two contexts. Raw query overlap alone does not establish attention collapse or harmful task crosstalk.',
        source_identity='A_t=rho_t+eps I; B=sum_t A_t; M_t=B^-1/2 A_t B^-1/2; chi=B^1/2 phi/sqrt(phi^dagger B phi). Alpha=chi^dagger M_t chi, sum_t M_t=I.',
        proven_bound='TV(alpha(phi),alpha(psi)) <= sqrt(1-|phi^dagger B psi|^2/(phi^dagger B phi psi^dagger B psi)). For fixed utilities: abs(logit(querychange)) <= utility_range*TV. Full40 margin certificate and CE upper bound follow coordinatewise.',
        novelty='Known POVM, pure-state trace distance and TV duality; not a new L5 contribution.',
        new_model_forwards=0,new_data_reads=0,new_training=0,new_formal_runs=0,l5_accepted=False)
    (P/'context-geometry-audit.json').write_bytes((json.dumps(output,indent=2)+'\n').encode())
    print(json.dumps(dict(status='PASS',raw_fidelity=float(raw),TV={k:v['illustrative_floats']['TV'] for k,v in result.items()},full40_checked=True,l5_accepted=False)))
if __name__=='__main__':main()
