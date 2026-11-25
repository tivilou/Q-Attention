from pathlib import Path
import hashlib,json,sys,time
r=Path(__file__).resolve().parent
assert not (r/'objective-diagnostic.json').exists(),'Reuse recorded results; no repeated diagnostic.'
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
cmd=[ops.PY,'-B','-'];code=(r/'diagnose.py').read_bytes();start=time.monotonic()
raw=ops.remote(cmd,data=code,timeout=60)
v=json.loads(raw);assert v['status']=='PASS_EXISTING_GEN98_OBJECTIVE_DIAGNOSTIC'
v['diagnostic_protocol_sha256']=hashlib.sha256((r/'diagnostic-protocol.json').read_bytes()).hexdigest()
v['diagnostic_code_sha256']=hashlib.sha256(code).hexdigest()
(r/'objective-diagnostic.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
(r/'execution.json').write_text(json.dumps(dict(remote_argv=cmd,stdin_code_sha256=v['diagnostic_code_sha256'],protocol_sha256=v['diagnostic_protocol_sha256'],literal_stdout=raw,literal_stderr='',exit_status=0,transport_elapsed_seconds=time.monotonic()-start,transport='existing bound ops.remote; nonzero remote status raises'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(seconds=v['seconds'],models={m:dict(brier=a['overall']['brier'],class_macro_brier=a['class_macro_brier'],ce=a['overall']['ce'],macro_f1=a['original_macro_f1'],high_wrong=a['high_confidence_wrong']['rows'],gradient_ratio=a['high_confidence_wrong']['mean_brier_to_ce_gradient_ratio']) for m,a in v['models'].items()},q_minus_g=v['q_minus_g_decomposition'])))
