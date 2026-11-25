from pathlib import Path
import difflib,hashlib,json,sys
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read('summary.json');v=read('verification.json');a=read('independent-verification.json');cfg=read('design.json')
assert s['decision']==a['decision']=='STOP_THIS_EXACT_ALIASING_PREDICTION'
assert s['gates']==dict(technical_numeric=True,population=True,main_auc_at_least_055=False,advantage_at_least_002=False)
assert v['state_before']==v['state_after'] and v['argmax_mismatches']==0 and v['sealed_macro_error']==0
assert read('collect-execution.json')['exit_status']==0 and read('transport-failed/collect-execution.json')['exit_status']==1
assert read('transport-failed/collect-execution.json')['code_sha256']==sha(r/'transport-failed/collect.py')
old=(r/'transport-failed/collect.py').read_text('utf8');new=(r/'collect.py').read_text('utf8')
assert new==old.replace("p.write_text(b,encoding='utf8')","p.write_bytes(b.encode('utf8'))")
(r/'collection-repair.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='collect.before.py',tofile='collect.py')),encoding='utf8')
for n,h in read('RUN_COMPLETE').items():assert sha(r/n)==h
for n,h in cfg['code_hashes'].items():assert sha(r/n)==h
assert sha(cfg['identity_spec']['prediction']['path'])==cfg['hypothesis_sha256']
sys.path.insert(0,'D:/ResearchInfrastructure/scheduler/runtime/17831b100d60915a')
from allocation_store import Allocations
c=json.loads(Path('D:/ResearchInfrastructure/scheduler/config.json').read_bytes());store=Allocations(c['database'])
launch=read('launch.json');row=next(x for x in store.rows() if x['allocation_id']==launch['allocation_id'])
assert row['generation']==105 and row['state']=='released' and row['runner_attempt_id']==launch['runner_attempt_id']
assert read('allocation-closeout.json')['state']=='released'
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code='from pathlib import Path;import json,sys;print(json.dumps({"original_owner_pid_absent":not Path("/proc",sys.argv[1]).exists()}))'
proc=json.loads(ops.remote([ops.PY,'-B','-c',code,str(launch['pid'])],timeout=15));assert proc['original_owner_pid_absent']
resource=dict(status='PASS_EXACT_GEN105_RETURN',allocation_id=row['allocation_id'],generation=row['generation'],runner_attempt_id=row['runner_attempt_id'],state='released',
 original_owner_pid_absent=True,new_capacity_purchased=False,new_COS_objects=0,model_API_calls=0,private_task_arrays_on_windows=False)
(r/'resource-verification.json').write_text(json.dumps(resource,indent=2)+'\n',encoding='utf8')
paths=[
 'D:/CodexHome/task-artifacts/qattn-observable-value-span-20261126/design.json',
 'D:/CodexHome/task-artifacts/qattn-observable-value-span-20261126/runner.py',
 'D:/CodexHome/task-artifacts/qattn-observable-value-span-20261126/protocol.md',
 'D:/CodexHome/task-artifacts/qattn-state-value-preflight-20261126/derivation.md',
 'D:/CodexHome/task-artifacts/qattn-state-value-preflight-20261126/history-comparison.json',
 'D:/CodexHome/task-artifacts/qattn-operator-valued-source-20261125/analysis.md']
for p in paths:assert Path(p).is_file(),p
next=dict(status='NEXT_SOURCE_BOUND_JOINT_STATE_VALUE_SCOPE_CHECK_ONLY',
 action='Read the pinned fixed-teacher value-span and same-state observable definitions to determine whether joint end-to-end learning of native word states and bounded observable utility is a genuinely untested task hypothesis. State a distinct mechanism/control/falsifiable prediction or NONE; no task computation now.',
 rationale='Gen102 disproved fixed-teacher value replacement under its frozen error/argmax gates, not every jointly learned state/value model. This logical scope gap is not positive efficacy evidence. Need exact exclusion of prior Q-BAM/connected-value/operator-valued work before any candidate admission.',
 source_pins=[dict(path=p,sha256=sha(p)) for p in paths],
 minimum='Preserve original quantum router and same-core goal; compare exact Hermitian64 technical representation and strongest trained Gaussian/classical controls, distinguish any proposed joint hypothesis from teacher imitation and known density-moment identities. Read quantum-value original sources before a novelty claim.',
 candidate_selected=False,training_admitted=False,resources_next=dict(GPU=0,new_task_forward=0,fit=0,new_capacity=0),
 stop='If already covered, input-independent, only a known moment/weak-value renaming, or merely a gen102 rank/ridge/teacher/seed rescue, record NONE. Do not reopen the stopped pairing statistic or entity-conditioned query route.',
 retained_stops=['gen98 fixed screen','gen102 fixed-teacher value replacement','gen104 fixed-C label direction','gen105 exact pairing prediction','all other exact stopped profiles'],
 explicitly_not_next='Entity-conditioned query already had a completed full study Qtest .521912 below strongest real .546448; no instance-query rebranding.',
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',original_L4_card_sha256='ba10c598e8f12b91668a339cff8d367647025124494afac0f37221c1f023d89e')
(r/'next-direction.json').write_text(json.dumps(next,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
out=dict(status='PASS_GEN105_NEGATIVE_DIAGNOSTIC_CLOSEOUT',decision=s['decision'],gates=s['gates'],auc=s['auc'],counts=s['counts'],
 exact_fraction_verification=a['status'],source_hypothesis_unchanged=True,collection_fix='Only byte-preserving transport, no task rerun or scientific change; first failed collection receipt and converted copies retained.',
 resources=resource,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'closeout.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],decision=s['decision'],allocation='released',next_candidate_selected=False)))
