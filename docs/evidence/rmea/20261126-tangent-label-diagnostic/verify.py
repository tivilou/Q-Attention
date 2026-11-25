from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read('design.json');s=read('summary.json');v=read('verification.json');a=read('output-audit.json');e=read('execution.json');res=read('resource-verification.json')
for n,h in d['code_hashes'].items():assert sha(r/n)==h
assert sha(r/'protocol.md')==d['protocol_sha256'] and sha(r/'runner.diff')==d['implementation_diff_sha256']
old=read('failed-v1/design.json');original=(r/'failed-v1/runner.py').read_text('utf8')
assert original.replace("assert len(metadata[i]['tokens'])==b-a","assert metadata[i]['words']==b-a")== (r/'runner.py').read_text('utf8')
for key in ('fixed_inputs','input_hashes','batch_size','wall_seconds','resources','expected','gate'):assert d[key]==old[key]
assert read('failed-v1/execution.json')['exit_code']==1 and "KeyError: 'tokens'" in read('failed-v1/execution.json')['stderr']
assert read('implementation-review.json')['observed_task_forwards']==0 and read('implementation-review.json')['observed_task_labels']==0
assert e['exit_code']==0 and e['stderr']=='' and e['runner_attempt_id']==read('launch.json')['runner_attempt_id']
assert s['status']=='COMPLETE' and s['decision']=='STOP_CURRENT_LABEL_DIRECTION' and not s['model_admission']
assert s['gates']==dict(numerical=True,counts=True,error_advantage=False,correct_harm=True)
assert s['native_head_forwards']==192 and s['native_batched_circuit_calls']==384 and s['optimizer_updates']==0 and s['task_rows']==12288
assert v['status']=='PASS' and v['state_before']==v['state_after'] and all(x<=1e-8 for x in v['errors'].values())
assert v['trace']['samples']==8 and v['trace']['representations']==192 and v['trace']['elements']==411179 and not v['trace']['warnings']
assert a['status']=='PASS_SAVED_OUTPUT_AUDIT' and a['error_advantage']==s['error_advantage']
assert a['trace_elements']==v['trace']['elements'] and a['raw_tokens_available_cases']==1
assert res['status']=='PASS_EXACT_ALLOCATION_RETURN' and all(x['state']=='released' for x in res['allocations'])
receipt=read('RUN_COMPLETE')
for n in ('summary','verification'):assert sha(r/(n+'.json'))==receipt[n+'_sha256']
for split,count in [('fit',8192),('assessment',4096)]:
    for row in s['metrics'][split].values():assert sum(c['n'] for c in row['per_gold_class'])==row['all']['n']==count
assert read('audit-execution.json')['exit_status']==0 and read('audit-execution.json')['code_sha256']==sha(r/'audit_code.py')
assert json.loads(read('audit-execution.json')['stdout'])==a
out=dict(status='PASS_FROZEN_DIAGNOSTIC_CLOSEOUT',frozen_design_sha256=sha(r/'design.json'),
    scientific_contract_unchanged_in_repair=True,failed_attempt_preserved=True,complete_attempt_verified=True,allocations_released=2,
    decision=s['decision'],error_advantage=s['error_advantage'],trace_scope=v['trace']['completeness'],original_raw_tokens_missing_cases=7,
    private_arrays_downloaded=False,new_L3_L4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'closeout-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
