from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
launch=json.loads((r/'launch.json').read_bytes())
v=json.loads(ops.remote(['/hy-tmp/Q-Attention/env/bin/python','-B','-c',(r/'verify_remote.py').read_text('utf8'),launch['runner_attempt_id']],timeout=40))
assert v['execution']['runner_attempt_id']==launch['runner_attempt_id']==v['resource_return']['runner_attempt_id']
assert v['resource_return']['allocation_id']==launch['allocation_id'] and v['resource_return']['generation']==launch['generation']
for n,b in v['files'].items():
 p=r/n;assert not p.exists();p.write_text(b,encoding='utf8')
for n,k in [('execution.json','execution'),('resource-return.json','resource_return'),('independent-verification.json','independent_verification')]:
 (r/n).write_text(json.dumps(v[k],indent=2)+'\n',encoding='utf8')
seal=json.loads((r/'RUN_COMPLETE').read_bytes())
for n,h in seal.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
s=json.loads((r/'summary.json').read_bytes());num=json.loads((r/'verification.json').read_bytes())
assert num['status']=='PASS' and v['independent_verification']['decision']==s['decision']
assert s['training_admitted'] is False and not s['method_selected']
print(json.dumps({k:s[k] for k in ('decision','auc','gates','counts','advantage_over_best_control','elapsed_seconds','head_forwards','native_batched_circuit_calls','optimizer_updates')}))
print(json.dumps({k:num[k] for k in ('sealed_logit_max_scaled_error','argmax_mismatches','real16_kernel_error','independent_numpy_max_error','sealed_macro_error')}))
