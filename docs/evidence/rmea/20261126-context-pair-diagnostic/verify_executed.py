from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
cfg=json.loads((r/'design.json').read_bytes());ex=json.loads((r/'execution.json').read_bytes())
for n,h in cfg['code_hashes'].items():assert sha((r/n).read_bytes())==ex['file_hashes'][n]==h
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code='from pathlib import Path;import json; r=Path("/hy-tmp/Q-Attention/control/context-pair-diagnostic-20261126");print(json.dumps({n:(r/n).read_text() for n in ("design.json","protocol.md")}))'
remote=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=15));out=[]
for name,text in remote.items():
 b=text.encode();local=(r/name).read_bytes();assert sha(b)==ex['file_hashes'][name]
 assert local.replace(b'\r\n',b'\n')==b
 if name.endswith('.json'):assert json.loads(b)==cfg
 p=r/('staged-'+name);assert not p.exists();p.write_bytes(b)
 out.append(dict(local=name,executed=p.name,local_sha256=sha(local),executed_sha256=sha(b),identical_semantics=True,only_newline_normalization=local!=b))
assert ex['inputs']==cfg['input_hashes'] and ex['exit_code']==0
out=dict(status='PASS_ACTUAL_EXECUTED_SOURCE_AND_CONFIG',configuration=out,source_hashes_exact=True,original_frozen_configuration_semantics_unchanged=True,
 literal_command=ex['argv'],literal_stdout=ex['stdout'],literal_stderr=ex['stderr'],exit_status=ex['exit_code'])
(r/'execution-verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],configuration=out['configuration'],source_hashes_exact=True)))
