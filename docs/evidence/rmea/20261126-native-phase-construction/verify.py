from pathlib import Path
import hashlib,json,math
import numpy as np
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
def unpack(x):
    a=np.asarray(x['real'])+1j*np.asarray(x['imag']) if 'real' in x else np.asarray(x['values'])
    assert list(a.shape)==x['shape']
    return a
cfg=read('scope.json');receipt=read('receipt.before-execution.json');result=read('results.json')
execution=read('construct-execution.json');stage=read('native-stage.json');trace=read('trace.json')
assert execution['exit_status']==0 and not execution['stderr']
assert execution['code_sha256']==receipt['code_sha256']['construct.py']==sha((r/'construct.py').read_bytes())
assert json.loads(execution['stdout'])==result
assert sha((r/'independent_reference.py').read_bytes())==receipt['code_sha256']['independent_reference.py']
assert sha((r/'scope.json').read_bytes())==receipt['scope_sha256']==result['scope_sha256']
assert sha((r/'source-pins.json').read_bytes())==receipt['source_pins_sha256']==result['source_pins_sha256']
for p in read('source-pins.json'):assert sha(Path(p['path']).read_bytes())==p['sha256']
assert sha((r/'native-stage.json').read_bytes())==result['native_stage_sha256']==trace['stage_sha256']
assert sha((r/'trace.json').read_bytes())==result['trace_sha256']
assert stage['parameter_digest_before']==stage['parameter_digest_after']==result['parameter_digest']
assert sum(stage['parameter_nonzero_counts'].values())==4
assert len(stage['calls']['head'])==2 and len(stage['calls']['circuit'])==4
assert [x['mode'] for x in stage['calls']['head']]==['quantum','dephased']
assert result['native_head_calls']==2 and result['native_circuit_calls']==4
assert cfg['exact_calls']['native_head_forwards']==2 and cfg['exact_calls']['native_circuit_calls']==4
assert all(v==0 for v in result['resources'].values())
h=unpack(stage['inputs']['h']);roles=unpack(stage['inputs']['roles']);mask=unpack(stage['inputs']['mask'])
assert h.shape==(2,2,1024) and roles.shape==(2,2,4) and mask.shape==(2,2)
assert not h[:,:,3:].any() and not roles.any() and mask.all()
assert np.array_equal(h[:,:,0],np.array(cfg['synthetic_input']['h0']))
assert np.array_equal(h[:,:,1],np.array(cfg['synthetic_input']['h1']))
assert np.array_equal(h[:,:,2],np.array(cfg['synthetic_input']['h2']))
Q={n:unpack(x) for n,x in stage['quantum'].items()};D={n:unpack(x) for n,x in stage['dephased'].items()}
g={n:unpack(x) for n,x in trace['guided'].items()};ref={n:unpack(x) for n,x in trace['independent'].items()}
tol=cfg['fixed']['tolerance'];eps=cfg['fixed']['EPS']
assert Q['values'].shape==(2,2,64) and Q['alpha'].shape==(2,40,2) and Q['psi'].shape==(2,2,8)
assert stage['quantum']['psi']['dtype']=='complex128' and stage['quantum']['values']['dtype']=='float64'
assert np.max(abs(Q['psi']-ref['psi']))==result['errors']['native_vs_dense_psi']
assert np.max(abs(g['B']-ref['B']))==result['max_numeric_error']
assert np.max(abs(Q['alpha'][0]-Q['alpha'][1]))==0
assert np.max(abs(Q['values'][0]-Q['values'][1]))==0
assert np.max(abs(abs(Q['psi'][0])**2-abs(Q['psi'][1])**2))==0
assert np.max(abs(Q['logits']))==np.max(abs(D['logits']))==0
assert np.max(abs(Q['alpha']-.5))<=tol
assert np.max(abs(Q['values'][:,:,0]-h[:,:,1]))==0
assert np.max(abs(Q['values'][:,:,1]-h[:,:,2]))==0 and not Q['values'][:,:,2:].any()
assert not Q['utilities'][:,1:].any()
expected_C=np.zeros((2,40,8,8),complex);expected_C[1,0,0,2]=expected_C[1,0,2,0]=-1
assert np.max(abs(g['C']-expected_C))<=tol
expected_B=2*eps*np.eye(8,dtype=complex);expected_B[0,0]+=1;expected_B[2,2]+=1
assert np.max(abs(g['B'][1]-expected_B))<=tol
assert np.max(abs(g['z0']))==0
expected_q=np.zeros((2,40,8),complex);expected_q[:,:,0]=1
expected_q[1,0,0]=math.cosh(1)/math.sqrt(math.cosh(2))
expected_q[1,0,2]=-math.sinh(1)/math.sqrt(math.cosh(2))
assert np.max(abs(g['q']-expected_q))<=tol
expected_score=math.tanh(2)/(1+2*eps)
assert abs(g['ratio'][1,0]-expected_score)<=tol and not g['ratio'][0].any() and not g['ratio'][1,1:].any()
assert all(result['gates'].values()) and result['CPU_seconds']<=10 and result['max_numeric_error']<=tol
assert result['status']=='PASS_NATIVE_SYNTHETIC_PHASE_CONSTRUCTION'
assert result['joint_native_reachability']=='EXPLICIT_SYNTHETIC_FIXTURE_ONLY'
assert not result['task_utility'] and not result['literature_novelty'] and not result['new_theory']
assert result['goal']=='ACTIVE' and result['newL3']==result['newL4T']=='NOT_ACCEPTED'
out=dict(status='PASS_NATIVE_CONSTRUCTION_RECORDS',claim='Explicit constructed head input/parameter reachability only',
         head_calls=2,circuit_calls=4,synthetic_contexts=2,max_numeric_error=result['max_numeric_error'],
         CPU_seconds=result['CPU_seconds'],guided_class0=result['guided_class0'],
         analytic_C_and_query_checks=True,complete_stage_and_guided_numeric_trace=True,
         native_reruns=0,scope_sha256=result['scope_sha256'],trace_sha256=result['trace_sha256'],
         original_gate_family_certified=False,task_utility=False,new_theory=False,
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
