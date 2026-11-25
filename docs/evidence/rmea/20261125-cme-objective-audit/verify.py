from pathlib import Path
import json,hashlib,math,sys,time
r=Path(__file__).resolve().parent
load=lambda name:json.loads((r/name).read_bytes())
v=load('objective-diagnostic.json');execution=load('execution.json')
assert execution['exit_status']==0 and not execution['literal_stderr']
expected=json.loads(execution['literal_stdout'])
assert all(v[k]==val for k,val in expected.items())
assert v['diagnostic_protocol_sha256']==hashlib.sha256((r/'diagnostic-protocol.json').read_bytes()).hexdigest()
assert v['diagnostic_code_sha256']==hashlib.sha256((r/'diagnose.py').read_bytes()).hexdigest()
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
assert not (r/'independent-diagnostic.json').exists(),'Reuse independently verified results.'
code=(r/'verify_remote.py').read_bytes();cmd=[ops.PY,'-B','-'];start=time.monotonic()
raw=ops.remote(cmd,data=code,timeout=60);ind=json.loads(raw)
(r/'independent-diagnostic.json').write_text(json.dumps(ind,indent=2)+'\n',encoding='utf8')
(r/'verification-execution.json').write_text(json.dumps(dict(remote_argv=cmd,stdin_code_sha256=hashlib.sha256(code).hexdigest(),literal_stdout=raw,literal_stderr='',exit_status=0,transport_elapsed_seconds=time.monotonic()-start),indent=2)+'\n',encoding='utf8')
errors=[];leaves=0
def compare(x,y,path):
    global leaves
    if isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:compare(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(a,b) in enumerate(zip(x,y)):compare(a,b,path+'/'+str(i))
    elif isinstance(x,float):
        assert math.isfinite(x) and math.isfinite(y)
        error=abs(x-y);assert error<1e-12,(path,x,y)
        errors.append(error);leaves+=1
    else:
        assert x==y,(path,x,y);leaves+=1
compare(v['models'],ind['models'],'models');compare(v['fit_support'],ind['fit_support'],'fit_support')
assert sum(a['rows'] for a in v['models']['quantum_distinct']['paired_groups'].values())==4096
decomp=v['q_minus_g_decomposition']
assert abs(decomp['brier']-decomp['probability_square_sum']-decomp['minus_two_truth_probability'])<1e-14
receipt=load('fulltexts/8f220cfb35184edc/receipt.json')
assert receipt['identity']=='verified' and receipt['observed_version']=='2403.10859v1'
assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert len(pages)==17 and '2403.10859v1' in pages[0]['text']
assert 'Theorem 3.1' in pages[3]['text'] and 'Proof of Theorem 3.1' in pages[11]['text']
out=dict(status='PASS_INDEPENDENT_EXISTING_PREDICTION_RECOMPUTATION',
    all5_arms_all40_class_and_all8_paired_groups_verified=True,compared_scalar_leaves=leaves,
    maximum_absolute_numeric_error=max(errors),max_brier_identity_residual=ind['max_brier_identity_residual'],
    original_confusion_hashes_equal=True,diagnostic_protocol_and_code_hashes_verified=True,
    source_identity_hash_and_scope_locators_verified=True,independent_cpu_seconds=ind['seconds'],
    new_head_forwards=0,new_updates=0,new_gpu_allocations=0,new_model_feature_text_reads=0,
    existing_assessment_rows=4096,independent_confirmation=False,external_scientific_review=False,
    gen98_13_gate_decision='UNCHANGED_STOP_EXACT_CONFIGURATION',
    new_method_selected=False,new_T='NOT_ESTABLISHED',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',goal='ACTIVE')
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out))
