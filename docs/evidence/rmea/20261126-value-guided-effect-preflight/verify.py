"""Reopen frozen receipts, literal executions, retained pins and bounded review."""
from pathlib import Path
import ast
import hashlib
import json

r=Path(__file__).resolve().parent


def sha(b): return hashlib.sha256(b).hexdigest()
def load(n): return json.loads((r/n).read_bytes())


for receipt in ('receipt.before-execution.json','contrast-receipt.before-execution.json'):
    assert all(sha((r/n).read_bytes())==h for n,h in load(receipt)['files'].items())
for filename in ('source-pins.json','contrast-source-pins.json'):
    for x in load(filename).values():
        assert sha(Path(x['path']).read_bytes())==x['sha256'],x['path']
scope=load('scope.json')
a,b,c=load('results.json'),load('independent-results.json'),load('contrast-results.json')
assert a['status']=='PASS_BOUNDED_EFFECT_ALGEBRA_CHECKS' and a['random_rows']==960 and a['algebra_evaluations']==3840
assert b['status']=='PASS_INDEPENDENT_EXACT_RATIONAL_EFFECT_CHECKS' and b['exact_native_moment_cases']==12
assert b['primary_results_sha256']==sha((r/'results.json').read_bytes())
assert c['status']=='PASS_RESIDUAL_GEOMETRY_FUNCTIONAL_CONTRAST'
assert c['context_A']['guided']==0 and abs(c['context_B']['guided']-.9640256520245128)<1e-14
assert c['real8_exact_for_this_real_fixture'] and not c['task_utility'] and not c['literature_novelty']
for stem,target in (('preflight','results.json'),('independent','independent-results.json'),('contrast','contrast-results.json')):
    q=load(stem+'-execution.json')
    assert q['exit_status']==0 and q['stderr']==''
    assert q['code_sha256']==sha((r/(stem+'.py')).read_bytes())
    answer=json.loads(q['stdout'])
    if stem!='contrast': assert answer==load(target)
    else: assert all(load(target)[k]==v for k,v in answer.items())
    imports=set()
    for node in ast.walk(ast.parse((r/(stem+'.py')).read_text('utf8'))):
        if isinstance(node,ast.Import): imports.update(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom): imports.add(node.module)
    assert imports<={'pathlib','hashlib','json','time','numpy','fractions'}
assert all(x==0 for x in scope['resource'].values())
assert a['nonmonotone_strength']['scores'][0]>a['nonmonotone_strength']['scores'][1]>0
case=a['forty_class_counterexample']
assert case['original_prediction']==case['gold']==0 and case['updated_prediction']==1
assert case['updated_CE']>case['original_CE'] and case['all_scores_baseline_nondecreasing']
history=load('history-comparison.json')
assert history['inherited_obstruction']['count']==431 and not history['task_method_selected']
review=load('review/review.json')
assert review['model']=='gpt-6.2-astra' and review['reasoning_effort']=='high' and review['exact_model_effort_echo_verified']
assert review['inference_requests']==1 and not review['task_arrays_sent'] and not review['tool_authority']
assert review['response']['static_verdict']=='PASS' and review['response']['next_action']=='ORIGINAL_SOURCE_AUDIT'
assert len(review['response']['must_fix'])==1 and not review['response']['GPU_admission'] and not review['response']['new_theory_accepted']
assert review['packet_sha256']==sha((r/'review/packet.json').read_bytes())
assert all(sha((r/n).read_bytes())==h for n,h in review['file_hashes'].items())
raw=load('review/response.private.json')
assert sha((r/'review/response.private.json').read_bytes())==review['raw_response_sha256']
assert raw['model']=='gpt-6.2-astra' and raw['reasoning']['effort']=='high'
old=(r/'derivation.md').read_text('utf8')
before='Strict increase holds for s>0 if Cphi is nonzero.'
after='If Cphi is nonzero, then z(q_s)>z0 for every finite s>0. This is a strict comparison with the starting score, not monotonicity in s.'
assert old.count(before)==1 and (r/'derivation-v2.md').read_text('utf8')==old.replace(before,after)
n=load('next-direction.json')
assert n['status']=='FROZEN_TWO_NEIGHBOR_ORIGINAL_SOURCE_AUDIT' and len(n['source_targets'])==2
assert not n['selected_task_method'] and not n['GPU_admission'] and all(x==0 for x in n['resource'].values())
assert all(x['verified_identifier'] is None for x in n['source_targets'])
assert n['goal']==a['goal']==c['goal']=='ACTIVE' and n['newL3']==n['newL4T']=='NOT_ACCEPTED'
out=dict(status='PASS_EFFECT_PREFLIGHT_AND_REVIEW_RECORDS',frozen_receipts_verified=True,literal_three_execution_exit0_verified=True,
         retained_source_pins_verified=True,review_model_effort_packet_echo_verified=True,review_required_wording_clarification_applied=True,
         preflight_resource=scope['resource'],separate_review_inference_requests=1,
         conditional_inherited_obstructions=431,selected_task_method=False,GPU_admission=False,
         external_static_mechanism_review='PASS_WITH_WORDING_CLARIFICATION_APPLIED',independent_task_confirmation=False,
         new_theory=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
