from pathlib import Path
import ast, hashlib, json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read('summary.json');p=read('source-pins.json');a=read('reading-receipt.json');n=read('next-direction.json')
assert len(p)==10 and all(sha(x['path'])==x['sha256'] for x in p)
assert s['decision']=='NONE_NEW_STRUCTURAL_CANDIDATE' and not s['candidate_selected'] and not s['training_admitted']
assert all(x==0 for x in s['resources'].values())
assert a['text_pages_read']==list(range(1,8)) and a['visually_inspected_pages']==[3,4,5,7]
assert a['unread_pages']==list(range(8,14))
receipt=read('fulltexts.private/011f390a050bf4d3/receipt.json')
assert receipt['identity']=='verified' and receipt['sha256']==a['pdf_sha256']==sha(receipt['path'])
assert receipt['version']==a['version']=='arxiv_v2'
assert n['resources_next']==dict(GPU=0,new_task_forward=0,fit=0,new_capacity=0)
assert not n['candidate_selected'] and not n['training_admitted']
assert all(sha(x['path'])==x['sha256'] for x in n['source_pins'])
assert len(read('comparison.json'))==6
assert '第8–13页' in (r/'analysis.md').read_text('utf8')
assert read('report-execution.json')['exit_status']==0
assert read('report-execution.json')['code_sha256']==sha(r/'report.py')
for name in ('report.py','verify.py','resolve_source.py','execute.py'):ast.parse((r/name).read_bytes())
out=dict(status='PASS_SOURCE_IDENTITY_SCOPE_AND_RETAINED_PINS',source_pins=10,
 paper_sha256=a['pdf_sha256'],reading_attestation='primary_scope_reading_not_external_review',
 source_conclusion=s['decision'],new_empirical_evidence=False,
 limitations=['Semantic reading is a primary attestation; hash checks do not independently prove interpretation.','No external novelty/theory review or task efficacy test occurred.'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
