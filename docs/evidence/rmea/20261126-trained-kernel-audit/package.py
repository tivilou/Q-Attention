from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
v=json.loads((r/'verification.json').read_bytes())
assert v['status']=='PASS_MECHANICAL_SOURCE_AND_TABLE_PACKAGE'
assert all(sha((r/n).read_bytes())==s for n,s in v['evidence_sha256'].items())
e=json.loads((r/'verification-execution.json').read_bytes())
assert e['exit_status']==0 and e['stderr']=='' and e['code_sha256']==sha((r/'verify.py').read_bytes())
o=json.loads(e['stdout'].strip());assert o['read_pages']==17 and o['source_table_rows']==6
assert o['status']==v['status'] and o['goal']=='ACTIVE'
names=tuple(v['evidence_sha256'])+('verify.py','run_verification.py','verification.json','verification-execution.json')
out=dict(status='PASS_SOURCE_READING_AND_EVIDENCE_PACKAGE',source_receipt_reading_verified=True,
         input_code_and_literal_exit_verified=True,mechanical_verification_only=True,
         public_evidence_sha256={n:sha((r/n).read_bytes()) for n in names},
         local_only_inputs=['PDF/parsed fulltext/renders/helper receipt/pre-reading backup, pinned native code already preserved elsewhere'],
         external_scientific_review=False,independent_task_confirmation=False,new_method_selected=False,
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],verified_public_evidence_files=len(names),literal_exit=0)))
