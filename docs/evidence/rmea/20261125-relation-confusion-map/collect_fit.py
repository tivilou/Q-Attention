"""Collect sealed fixed-fit aggregates, retaining all four existing models."""
from pathlib import Path
import hashlib
import json
import sys

r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
code=r'''from pathlib import Path
import hashlib,json
r=Path('/hy-tmp/Q-Attention/control/relation-confusion-map-20261125')
e=json.loads((r/'execution.json').read_bytes());ev=json.loads((r/'resource-return.json').read_bytes())
assert e['exit_code']==0 and ev['campaign_status']=='completed' and ev['quiescent'] and ev['no_remaining_consumers']
seal=(r/'output/RUN_COMPLETE').read_bytes();hs=json.loads(seal);assert set(hs)=={'summary.json'}
p=r/'output/summary.json';b=p.read_bytes();assert not p.is_symlink() and len(b)<512*1024
assert hashlib.sha256(b).hexdigest()==hs['summary.json']
print(json.dumps(dict(files={'summary.json':b.decode(),'RUN_COMPLETE':seal.decode()},execution=e,return_event=ev)))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=30))
launch=json.loads((r/'launch.json').read_bytes());cfg=json.loads((r/'design.json').read_bytes())
assert v['execution']['runner_attempt_id']==v['return_event']['runner_attempt_id']==launch['runner_attempt_id']
for n,text in v['files'].items():
    p=r/n
    if p.exists():assert p.read_bytes()==text.encode()
    else:p.write_bytes(text.encode())
for n,k in [('execution.json','execution'),('resource-return.private.json','return_event')]:
    (r/n).write_bytes((json.dumps(v[k],indent=2)+'\n').encode())
s=json.loads((r/'summary.json').read_bytes())
assert s['attempt']==launch['runner_attempt_id'] and s['rows']==8192 and s['forward_rows']==33280
assert s['design_sha256']==hashlib.sha256((r/'design.json').read_bytes()).hexdigest()
assert s['code_hashes']==cfg['code_hashes'] and s['input_hashes']==cfg['input_hashes']
assert s['optimizer_updates']==s['cos_writes']==0 and not s['valid_evaluated'] and not s['test_evaluated']
assert s['checkpoint_states_unchanged'] and set(s['results'])==set(cfg['models'])
assess=json.loads((r/'confusion-map.json').read_bytes());support=[x['fit_count'] for x in assess['classes']]
maximum=0.
for name,row in s['results'].items():
    cm=row['confusion_matrix'];assert len(cm)==40 and all(len(x)==40 for x in cm)
    assert all(type(v) is int and v>=0 for x in cm for v in x) and sum(map(sum,cm))==8192
    assert list(map(sum,cm))==row['class_support']==support
    f=[2*cm[i][i]/(sum(cm[i])+sum(x[i] for x in cm)) if sum(cm[i])+sum(x[i] for x in cm) else 0. for i in range(40)]
    maximum=max(maximum,abs(sum(f)/40-row['macro']),max(abs(x-y) for x,y in zip(f,row['per_class_f1'])))
    assert row['correct']==sum(cm[i][i] for i in range(40))
    assert sum(map(sum,row['positive39_confusion']))==sum(support[1:])==2792
    assert row['positive_forced_correct']==sum(row['positive39_confusion'][i][i] for i in range(39))
    assert row['assessment128_argmax_equal'] and row['assessment128_parity_max_error']<=cfg['parity_atol']
assert maximum<1e-12
record=dict(status='PASS',max_macro_error=maximum,all8192_fit_rows=True,all40_classes=True,all39_by39=True,
    all_four_stored128_argmax_equal=True,checkpoint_bytes_unchanged=True,optimizer_updates=0,
    no_new_valid_test=True,no_cos=True,no_windows_weights_or_task_arrays=True)
(r/'fit-verification.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(dict(seconds=s['seconds'],fit_macro={k:v['macro'] for k,v in s['results'].items()},
    parity_error={k:v['assessment128_parity_max_error'] for k,v in s['results'].items()},verification=record)))
