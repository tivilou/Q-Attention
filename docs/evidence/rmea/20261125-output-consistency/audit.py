"""Independent pure-Python 40-class metric and frozen gate recomputation."""
from pathlib import Path
import hashlib,json,math
r=Path(__file__).resolve().parent
read=lambda name:json.loads((r/name).read_bytes())
cfg=read('design.json');s=read('summary.json');e=read('execution.json');remote=read('remote-evidence.json')
marker=read('RUN_COMPLETE');launch=read('launch.json');fixture=read('fixtures.json')
assert e['exit_code']==0 and e['runner_attempt_id']==launch['runner_attempt_id']==s['attempt']
assert s['status']=='completed' and not s['valid_read'] and not s['test_read'] and s['cos_writes']==0
assert (s['train_items'],s['assessment_items'])==(8192,4096)
assert cfg['physical_optimizer_updates']==4096
for name,h in marker.items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h
assert s['code_hashes']==cfg['code_hashes']==remote['code_hashes']
assert s['source_hashes']==cfg['source_hashes']
assert s['design_sha256']==hashlib.sha256((r/'design.json').read_bytes()).hexdigest()
assert remote['input_hashes']==cfg['input_hashes']
assert all(remote['reference_row_matches'].values())
assert fixture['status']=='PASS' and all(fixture['cross_device_seed13_draws_equal'])
rows=dict(s['results'],**{'reused/'+k:v for k,v in s['reused_results'].items()})
assert len(rows)==8 and list(s['results'])==cfg['modes']
scores={};errors={}
for name,row in rows.items():
    cm=row['confusion_matrix'];assert len(cm)==40 and all(len(v)==40 for v in cm)
    assert all(isinstance(x,int) and x>=0 for v in cm for x in v)
    support=[sum(v) for v in cm];assert sum(support)==4096 and min(support)>0
    assert support==row['class_support']
    scores[name]=sum(2*cm[i][i]/(support[i]+sum(v[i] for v in cm)) for i in range(40))/40
    errors[name]=abs(scores[name]-row['all40_macro_f1'])
    assert errors[name]<1e-14 and row['updates']==1024 and row['parameters'] in (81028,82308)
    assert row['order_sha256']==cfg['reused_native_rows']['quantum']['order_sha256']
    assert support==cfg['reused_native_rows']['quantum']['class_support']
for name,row in s['results'].items():
    assert row['parameters']==81028 and row['objective_components']['updates']==1024
    assert row['compute_seconds']<600
    arms=remote['arms'][name]
    assert arms['train_seconds']<600
    assert math.isfinite(row['objective_components']['ce']) and math.isfinite(row['objective_components']['kl'])
assert min(v['assessment_started_at'] for v in s['results'].values())>=max(v['training_finished_at'] for v in s['results'].values())
gates={'macro_beats_'+k:scores['quantum_vat']>v for k,v in scores.items() if k!='quantum_vat'}
assert len(gates)==7 and gates==s['gates']
expected='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_OUTPUT_CONSISTENCY_SCREEN'
assert s['gate']==expected
record=dict(status='PASS',scope='Independent CM/gate, immutable input/code/source/order and temporal checks, not independent replication',
    macro_scores=scores,absolute_metric_errors=errors,gates=gates,gate=s['gate'],
    optimizer_updates=sum(v['updates'] for v in s['results'].values()),
    train_seconds_sum=sum(v['train_seconds'] for v in remote['arms'].values()),
    compute_seconds_sum=sum(v['compute_seconds'] for v in s['results'].values()),
    exact_reused_rows_verified=remote['reference_row_matches'],all_fits_before_any_assessment=True,
    actual8_and_cuda_pairing='PASS',cos_writes=0,new_valid_test_reads=False,
    L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
(r/'result-verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
