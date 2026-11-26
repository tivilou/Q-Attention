from pathlib import Path
from fractions import Fraction
import hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes());sha=lambda n:hashlib.sha256((r/n).read_bytes()).hexdigest()
s=read('summary.json');cfg=read('design.json');v=read('validation.json');e=read('execution.json');a=read('launch.json')
assert e['exit_code']==0 and s['attempt']==a['runner_attempt_id'] and s['status']=='completed'
assert s['train_items']==8192 and s['assessment_items']==4096 and not s['test_read'] and not s['valid_read'] and s['cos_writes']==0
assert s['design_sha256']==sha('design.json')
for n,h in cfg['code_hashes'].items():assert sha(n)==h
assert s['code_hashes']==cfg['code_hashes'] and s['source_hashes']==cfg['source_hashes']
assert v['real_task_rows']==8 and len(v['reconstructed_reference_initial_identities'])==8
def metric(row):
 cm=row['confusion_matrix'];assert len(cm)==40 and all(len(x)==40 for x in cm)
 assert all(type(x)==int and x>=0 for a in cm for x in a)
 support=[sum(x) for x in cm];pred=[sum(cm[j][i] for j in range(40)) for i in range(40)]
 assert sum(support)==4096 and support==cfg['assessment_class_support']
 f1=[Fraction(2*cm[i][i],support[i]+pred[i]) if support[i]+pred[i] else Fraction(0) for i in range(40)]
 macro=sum(f1)/40
 assert abs(float(macro)-row['all40_macro_f1'])<1e-14
 assert row['updates']==1024 and row['order_sha256']==cfg['order_sha256']
 return macro
fractions={k:metric(row) for k,row in s['results'].items()}
refs={k:metric(row) for k,row in cfg['references'].items()}
assert max(x['training_finished_at'] for x in s['results'].values())<=min(x['assessment_started_at'] for x in s['results'].values())
assert set(row['initial_sha256'] for row in s['results'].values())=={cfg['expected_initial_sha256']}
assert all(row['parameters']==81028 for row in s['results'].values())
q=fractions['quantum_middle']
gates={'Q_beats_'+k:q>x for k,x in fractions.items() if k!='quantum_middle'}
gates.update({'Q_beats_reference/'+k:q>x for k,x in refs.items()})
assert gates==s['gates']
expected='PASS_FOR_FULL_STUDY_PROTOCOL_ONLY' if all(gates.values()) else 'STOP_EXACT_MIDDLE_PLACEMENT_PROFILE'
assert s['gate']==expected
for n,h in read('RUN_COMPLETE').items():assert sha(n)==h
out=dict(status='PASS_INDEPENDENT_SCREEN_AUDIT',gate=expected,gates=gates,pass_count=sum(gates.values()),gate_count=len(gates),
 exact_all40_macro={k:str(x) for k,x in fractions.items()},exact_reference_macro={k:str(x) for k,x in refs.items()},
 independently_recomputed_metrics='Fraction from all40 integer confusion matrices; includes zero F1 where denominator zero',
 final_only=True,all_fits_before_assessment=True,shared_initial_full_state_verified=True,
 GPU_training_updates=3072,reference_refits=0,new_capacity=0,heldout_reads=False,
 newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',originalL4='RETAINED',goal='ACTIVE',
 limitations='Primary independent arithmetic audit; no independent new task replication. Exposed TRAIN development only. Null-coordinate and geometry/effective-capacity differences retained.')
(r/'verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps({k:out[k] for k in ('status','gate','pass_count','gate_count','newL3','newL4T')}))
